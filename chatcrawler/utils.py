import io
import secrets
import string

import PyPDF2
import requests
from langdetect import detect
import openai
from chatcrawler.chat import answer_question


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


def answer_arabic(question, df, answer_func):
    """
    Translates an Arabic question into English, passes it to answer_func to get the answer,
    and translates the answer back to Arabic. Returns the answer in Arabic and any related URLs.

    :param question: A string representing the Arabic question to be answered.
    :param df: A pandas DataFrame containing the data to be used to answer the question.
    :param answer_func: A function that takes the DataFrame and the English version of the question
                        and returns the answer and related URLs.
    :return: A tuple containing the answer to the question in Arabic and any related URLs.
    """

    q_english = translate_to_english(question)
    answer, urls = answer_func(df, question=q_english, debug=False)
    a_arabic = translate_to_arabic(answer)
    return a_arabic, urls


def general_answer(question, df):
    """
    Given a question and a dataframe, returns an answer and URLs related to the question.

    :param question: a string representing the question to answer.
    :param df: a pandas DataFrame containing the data to answer the question.
    :return: a tuple containing the answer and a list of URLs related to the question.
    """

    if is_arabic(question):
        answer, urls = answer_arabic(question, df, answer_question)
    else:
        answer, urls = answer_question(df, question=question, debug=False)

    return answer, urls
