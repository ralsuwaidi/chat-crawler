import gbot.utils.utils as utils
import openai
from openai.embeddings_utils import distances_from_embeddings


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

    q_english = utils.translate_to_english(question)
    answer, urls = answer_func(df, question=q_english, debug=False)
    a_arabic = utils.translate_to_arabic(answer)
    return a_arabic, urls


def create_context(question, df, max_len=1800, size="ada"):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(
        input=question, engine="text-embedding-ada-002"
    )["data"][0]["embedding"]

    # Get the distances from the embeddings
    df["distances"] = distances_from_embeddings(
        q_embeddings, df["embeddings"].values, distance_metric="cosine"
    )

    urls = set([])
    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values("distances", ascending=True).iterrows():
        # Add the length of the text to the current length
        cur_len += row["n_tokens"] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])
        urls.add(row["url"])

    # Return the context
    return "\n\n###\n\n".join(returns), sorted(urls, reverse=True)


def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context, urls = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f'Answer the question based on the context below, and if the question can\'t be answered based on the context, say "I could not find the answer in the databse but this resource may help"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:',
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip(), urls
    except Exception as e:
        print(e)
        return ""


class Answer:
    def answer(self, question: str, show_debug: bool = False):
        """
        Given a question, returns an answer along with relevant URLs.
        :param question: A string representing the question to be answered.
        :return: A tuple containing the answer as a string and a list of relevant URLs.
        """

  
        answer, urls = answer_question(self.df, question=question, debug=show_debug)

        return answer, urls
