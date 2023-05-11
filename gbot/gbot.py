import concurrent.futures
import os
import ssl
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from urllib.parse import urlparse
from gbot.answer import Answer

import numpy as np
import openai
import pandas as pd
import requests
import tiktoken
from bs4 import BeautifulSoup

from gbot.hyperlink import get_domain_hyperlinks
from gbot.logger import logger
import gbot.utils as utils

ssl._create_default_https_context = ssl._create_unverified_context


IGNORE_EXTENSION = (
    ".png",
    ".jpg",
    ".jpeg",
    ".xlsx",
    ".xls",
    ".docx",
    ".doc",
    ".xml",
)


class GBot(Answer):
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

    def worker(self):
        while True:
            try:
                url: str = self.queue.get()
            except Queue.Empty:
                logger.warning(f"queue is empty")
                break

            if url is None:
                break

            filename = os.path.join("text", self.domain, utils.generate_name())

            try:
                # Save text from the url to a <url>.txt file
                if not os.path.exists(filename):
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(f"{url}\n")
                        # Get the text from the URL using BeautifulSoup
                        response = self.session.get(url, verify=False)
                        content_type = response.headers.get("Content-Type")

                        # write pdf instead of saving as text
                        if content_type and "pdf" in content_type.lower():
                            text = utils.read_pdf_from_url(url)
                            logger.info(f"writing text from pdf {url}")
                            f.write(text)
                            self.queue.task_done()
                            continue

                        soup = BeautifulSoup(response.text, "html.parser")
                        text = soup.get_text()

                        # If the crawler gets to a page that requires JavaScript, it will stop the crawl
                        if "You need to enable JavaScript to run this app." in text:
                            self.queue.task_done()
                            logger.warn(f"skipping url due to js {url}")
                            continue

                        # Otherwise, write the text to the file in the text directory
                        logger.info(f"writing text {url}")
                        f.write(text)
                else:
                    logger.info(f"skipping file since it exists already {url}")

            except Exception as e:
                logger.error(f"Error writing to file {filename}: {e}")

            # Get the hyperlinks from the URL and add them to the queue
            for link in get_domain_hyperlinks(self.domain, url, self.all_sub_websites):
                if link not in self.seen:
                    # dont scrape these

                    if not (
                        link.endswith(IGNORE_EXTENSION)
                        or any(word in link for word in self.ignore_words)
                    ):
                        logger.info(f"adding link {link}")
                        self.queue.put(link)
                    else:
                        logger.info(f"skipping {link}")
                    self.seen.add(link)

            self.queue.task_done()

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

    def process_folder(self, folder):
        """
        Returns a processed dataframe from the text files in the given folder.

        :param folder: A string representing the path to the folder containing text files.
        :return: A pandas DataFrame with two columns: "url" and "text".

        This function reads the text files in the given folder, omits the first 11 lines and the last 4 lines,
        and replaces "-", "_", and "#update" with spaces. The resulting text is then stored in a list of tuples
        with the URL of the file. Finally, a DataFrame is created from the list and returned with the "text"
        column having the newlines removed.
        """

        # Create a list to store the text files
        texts = []

        # Get all the text files in the text directory
        for file in os.listdir(folder):
            # Open the file and read the text
            with open(folder + "/" + file, "r") as f:
                url = f.readline()
                text = f.read()

                # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
                texts.append((url[:-1], text))

        # Create a dataframe from the list of texts
        df = pd.DataFrame(texts, columns=["url", "text"])
        # Set the text column to be the raw text with the newlines removed
        df["text"] = utils.remove_newlines(df.text)
        self.df = df

        self.prepare_folder()
        self.df.to_csv(f"processed/processed_{self.domain}.csv", escapechar="\\")

    def shorten(self):
        """
        Shorten the text in a Pandas dataframe to chunks of a maximum number of tokens using cl100k_base.

        Returns:
        None.
        """

        # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
        tokenizer = tiktoken.get_encoding("cl100k_base")

        self.df.columns = ["url", "text"]
        # Tokenize the text and save the number of tokens to a new column
        self.df.dropna(subset=["text"], inplace=True)
        self.df["n_tokens"] = self.df.text.apply(lambda x: len(tokenizer.encode(x)))
        print(f"finished adding tokens")

        # Function to split the text into chunks of a maximum number of tokens
        shortened = []

        # Loop through the dataframe
        for row in self.df.iterrows():
            # If the text is None, go to the next row
            if row[1]["text"] is None:
                continue

            # If the number of tokens is greater than the max number of tokens, split the text into chunks
            if row[1]["n_tokens"] > self.max_tokens:
                shortened_chunks = utils.split_into_many(row[1]["text"])
                for i in shortened_chunks:
                    shortened.append([row[1]["url"], i])

            # Otherwise, add the text to the list of shortened texts
            else:
                shortened.append([row[1]["url"], row[1]["text"]])

        self.df = pd.DataFrame(shortened, columns=["url", "text"])
        self.df["n_tokens"] = self.df.text.apply(lambda x: len(tokenizer.encode(x)))

        self.df = self.df.drop(self.df[self.df["n_tokens"] < 2].index)

        print("finished shortening")

    def embed(self):
        """
        This function generates embeddings for a given list of texts using OpenAI's Text Embedding Ada 002
        engine. The function uses multithreading to create embeddings in parallel for each text.

        :return: a list of embeddings for each text in the input list
        """

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
