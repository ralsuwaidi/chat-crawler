import concurrent.futures
import os
import ssl
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from urllib.parse import urlparse

import numpy as np
import openai
import pandas as pd
import requests

from gbot.answer import Answer
from gbot.process import Process
from gbot.utils.logger import logger

ssl._create_default_https_context = ssl._create_unverified_context


class GBot(Answer, Process):
    def __init__(self, url: str = None):
        self.url = url
        self.domain = urlparse(self.url).netloc or None
        self.all_sub_websites = False
        self.max_filename_char = 200
        self.max_tokens = 500
        self.session = requests.Session()
        self.queue = Queue()
        # Create a set to store the URLs that have already been seen (no duplicates)
        self.seen = set([url])
        self.ignore_words = ["email-protection", "Login?Returnlink", "callback"]
        self.df = None

    def crawl(self):
        # Parse the URL and get the domain
        if not self.url:
            raise ValueError("url cannot be empty")

        self.domain = urlparse(self.url).netloc
        logger.info(f"scraping {self.domain}")

        # Create a queue to store the URLs to crawl
        self.queue.put(self.url)

        self.prepare_folder()

        with ThreadPoolExecutor() as executor:
            # Submit worker tasks to the executor

            logger.info(f"number of workers: {executor._max_workers}")

            futures = [
                executor.submit(self.worker) for _ in range(executor._max_workers)
            ]

            # Wait for all tasks to be completed
            self.queue.join()

            # Stop workers by putting None into the queue for each worker
            for _ in range(executor._max_workers):
                self.queue.put(None)

            # Wait for all worker threads to finish
            for future in futures:
                future.result()

    def __del__(self):
        self.session.close()

    @classmethod
    def from_embedding(cls, file) -> "GBot":
        df = pd.read_csv(file, index_col=0)
        df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)

        gbot = cls()
        gbot.df = df

        return gbot

    def to_embedding(self, folder):
        """
        This function generates embeddings for a given list of texts using OpenAI's Text Embedding Ada 002
        engine. The function uses multithreading to create embeddings in parallel for each text.

        :return: a list of embeddings for each text in the input list
        """

        # get text
        self.process_folder(folder)
        # shorten and tokenize the text and update self.df
        self.shorten()

        embeddings = []

        def create_embedding(text):
            return openai.Embedding.create(input=text, engine="text-embedding-ada-002")[
                "data"
            ][0]["embedding"]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i, result in enumerate(executor.map(create_embedding, self.df["text"])):
                embeddings.append(result)
                logger.info("Embedded %d/%d texts", i + 1, len(self.df))

        self.df["embeddings"] = embeddings
        filename = f"processed/embeddings_{self.domain}.csv"
        self.df.to_csv(filename, escapechar="\\")
        print("finished embedding", self.domain)
        print("saved embeddings to", filename)

    def prepare_folder(self):
        """
        Creates a directory to store text files for the given domain and a directory to store processed csv files.
        :return: None
        """

        # Create a directory to store the text files
        text_dir = "text/" + self.domain + "/"
        if not os.path.exists(text_dir):
            os.makedirs(text_dir)

        # Create a directory to store the csv files
        if not os.path.exists("processed"):
            os.mkdir("processed")
