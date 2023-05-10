import requests

from bs4 import BeautifulSoup

from chatcrawler.logger import logger

from urllib.parse import urlparse
import threading
from queue import Queue
import os
import ssl
from chatcrawler.hyperlink import get_domain_hyperlinks
from chatcrawler.utils import read_pdf_from_url

ssl._create_default_https_context = ssl._create_unverified_context

class Crawler:

    def __init__(self, url: str):
        self.url = url
        self.all_sub_websites = False
        self.max_filename_char = 200
        self.session = requests.Session()
        self.queue = Queue()
        # Create a set to store the URLs that have already been seen (no duplicates)
        self.seen = set([url])

    def worker(self, local_domain: str):
        while True:
            try:
                url: str = self.queue.get()
            except Queue.Empty:
                logger.warn(f"que is empty")
                break

            if url is None:
                break

            # shorten name
            url_no_https = url[8:]
            shortened_url = url_no_https[:self.max_filename_char]

            # filename = (
            #     "text/" + local_domain + "/" + shortened_url.replace("/", "_") + ".txt"
            # )

            filename = os.path.join("text", local_domain, shortened_url.replace("/", "_") + ".txt")


            try:
                # Save text from the url to a <url>.txt file
                if not os.path.exists(filename):
                    with open(filename, "w", encoding="utf-8") as f:

                        # Get the text from the URL using BeautifulSoup
                        logger.info(f"getting the url {url}")
                        response = self.session.get(url, verify=False)
                        content_type = response.headers.get('Content-Type')

                        # write pdf instead of saving as text
                        if content_type and 'pdf' in content_type.lower():
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
                    logger.info(f'skipping file since it exists already {url}')

            except Exception as e:
                logger.error(f"Error writing to file {filename}: {e}")

            # Get the hyperlinks from the URL and add them to the queue
            for link in get_domain_hyperlinks(local_domain, url, self.all_sub_websites):
                if link not in self.seen:
                    logger.info(f'adding link {link}')
                    # dont scrape these
                    if not (link.endswith(".jpeg") or link.endswith(".jpg") or "email-protection" in link or 'Login?Returnlink' in link or 'callback' in link):
                        self.queue.put(link)
                    else:
                        logger.info(f'skipping {link}')
                    self.seen.add(link)

            self.queue.task_done()


    def crawl(self):
        # Parse the URL and get the domain
        local_domain = urlparse(self.url).netloc

        # Create a queue to store the URLs to crawl
        self.queue.put(self.url)

        # Create a directory to store the text files
        text_dir = "text/" + local_domain + "/"
        if not os.path.exists(text_dir):
            os.makedirs(text_dir)

        # Create a directory to store the csv files
        if not os.path.exists("processed"):
            os.mkdir("processed")

        # Create worker threads
        num_worker_threads = 12
        threads = []
        for i in range(num_worker_threads):
            t = threading.Thread(target=self.worker, args=(local_domain))
            t.start()
            threads.append(t)

        # Wait for all tasks to be completed
        self.queue.join()

        # Stop workers
        for i in range(num_worker_threads):
            self.queue.put(None)
        for t in threads:
            t.join()


    def __del__(self):
        self.session.close()