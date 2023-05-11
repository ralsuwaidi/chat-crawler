import os
import ssl
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from openai.embeddings_utils import cosine_similarity, distances_from_embeddings

from chatcrawler.hyperlink import get_domain_hyperlinks
from chatcrawler.logger import logger
from chatcrawler.utils import generate_name, read_pdf_from_url

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


class Crawler:
    def __init__(self, url: str = None):
        self.url = url
        self.all_sub_websites = False
        self.max_filename_char = 200
        self.session = requests.Session()
        self.queue = Queue()
        # Create a set to store the URLs that have already been seen (no duplicates)
        self.seen = set([url])
        self.ignore_words = ["email-protection", "Login?Returnlink", "callback"]
        self.df_embedded = None

    def worker(self, local_domain: str):
        while True:
            try:
                url: str = self.queue.get()
            except Queue.Empty:
                logger.warning(f"queue is empty")
                break

            if url is None:
                break

            filename = os.path.join("text", local_domain, generate_name())

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
                            text = read_pdf_from_url(url)
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
            for link in get_domain_hyperlinks(local_domain, url, self.all_sub_websites):
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
        local_domain = urlparse(self.url).netloc
        logger.info(f"scraping {local_domain}")

        # Create a queue to store the URLs to crawl
        self.queue.put(self.url)

        # Create a directory to store the text files
        text_dir = "text/" + local_domain + "/"
        if not os.path.exists(text_dir):
            os.makedirs(text_dir)

        # Create a directory to store the csv files
        if not os.path.exists("processed"):
            os.mkdir("processed")

        with ThreadPoolExecutor() as executor:
            # Submit worker tasks to the executor

            logger.info(f"number of workers: {executor._max_workers}")

            futures = [
                executor.submit(self.worker, local_domain)
                for _ in range(executor._max_workers)
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

    def get_embedded_data(self, file):
        df = pd.read_csv(file, index_col=0)
        df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)
        self.df_embedded = df
