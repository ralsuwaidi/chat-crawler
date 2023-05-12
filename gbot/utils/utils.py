import io
import secrets
import string

import PyPDF2
import requests
from langdetect import detect
import openai

import tiktoken


# Function to get the contents of a PDF from a URL
def read_pdf_from_url(url: str) -> str:
    """
    Reads a PDF from a given url and returns its text content.

    Args:
        url (str): The url of the PDF to be read.

    Returns:
        str: The text content of the PDF.
    """

    response = requests.get(url, verify=False)
    content = io.BytesIO(response.content)
    reader = PyPDF2.PdfReader(content)
    text = ""
    for i in range(len(reader.pages)):
        text += reader.pages[i].extract_text()
    return text


def remove_newlines(serie):
    """
    Replaces newlines with spaces and removes double spaces from a given Pandas series.

    :param serie: A Pandas series where newlines and double spaces will be removed.
    :type serie: pandas.Series

    :return: The input series with newlines replaced by spaces and double spaces removed.
    :rtype: pandas.Series
    """

    serie = serie.str.replace("\n", " ")
    serie = serie.str.replace("\\n", " ")
    serie = serie.str.replace("  ", " ")
    serie = serie.str.replace("  ", " ")
    return serie


def generate_name():
    """
    Generates a random string of six alphanumeric characters using the secrets module.

    Returns:
        A string of six randomly generated alphanumeric characters.
    """

    alphabet = string.ascii_letters + string.digits
    random_string = "".join(secrets.choice(alphabet) for i in range(6))
    return random_string + ".txt"


def is_arabic(text: str):
    """
    Generates a random string of six alphanumeric characters using the secrets module.

    Returns:
        A string of six randomly generated alphanumeric characters.
    """

    return detect(text) == "ar"


def translate_to_english(text):
    """
    Translates the given `text` to English using OpenAI's language model.

    Args:
        text (str): The text to be translated to English.

    Returns:
        str: The translated text in English.
    """

    response = openai.Completion.create(
        prompt=f"translate this text to english:\n{text}\n",
        temperature=0,
        max_tokens=1800,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        model="text-davinci-003",
    )
    return response["choices"][0]["text"].strip()


def translate_to_arabic(text):
    """
    Translates the given `text` to Arabic using OpenAI's language model.

    Args:
        text (str): The text to be translated to Arabic.

    Returns:
        str: The translated text in Arabic.
    """

    response = openai.Completion.create(
        prompt=f"translate this text to arabic:\n{text}\n",
        temperature=0,
        max_tokens=1800,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        model="text-davinci-003",
    )
    return response["choices"][0]["text"].strip()


def split_into_many(text, max_tokens=500):
    """
    Splits a given text into multiple chunks of sentences, where each chunk has a maximum number of tokens
    defined by the max_tokens parameter. Uses the Hugging Face Tokenizer to encode the text and calculate
    the number of tokens. Returns a list of strings, where each string is a chunk of sentences.

    Args:
        text (str): The text to be split into chunks.
        max_tokens (int): The maximum number of tokens allowed in each chunk. Defaults to 500.

    Returns:
        list: A list of strings, where each string is a chunk of sentences.
    """

    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Split the text into sentences
    sentences = text.split(". ")

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):
        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks


def create_embedding(text):
    """
    Creates an embedding for the given text using the OpenAI API.

    Args:
        text (str): The input text to be embedded.

    Returns:
        list: A list of floats representing the embedding of the text.
    """

    return openai.Embedding.create(input=text, engine="text-embedding-ada-002")["data"][
        0
    ]["embedding"]
