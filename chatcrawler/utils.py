import io
import PyPDF2
import requests


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
