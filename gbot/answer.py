import gbot.utils as utils


class Answer:
    def answer(self, question: str, show_debug: bool = False):
        """
        Given a question, returns an answer along with relevant URLs.
        :param question: A string representing the question to be answered.
        :return: A tuple containing the answer as a string and a list of relevant URLs.
        """

        if utils.is_arabic(question):
            print("arabic")
            answer, urls = utils.answer_arabic(question, self.df, utils.answer_question)
        else:
            answer, urls = utils.answer_question(
                self.df, question=question, debug=show_debug
            )

        return answer, urls
