"""
the default process
"""

import re
from typing import Callable


# it provides a default function to extract the question from a raw string
def default_question_extractor(question: str, answer: str) -> str:
    """
    Extracts the default question from a given string.

    Parameters:
    q (str): The input string containing the question.

    Returns:
    str: The extracted default question.

    """
    result = question
    result = result.replace(answer, "", 1)
    return result


# it provides a default function to extract the question from a raw string
def default_answer_extractor(answer: str, question: str) -> str:
    """
    Extracts the default value of 'a'.

    Parameters:
        a (str): The input string.

    Returns:
        str: The extracted value of 'a'.
    """
    answer = answer.replace(question, "", 1)
    answer = answer.replace("Assistant:", "", 1)
    answer = answer.replace("assistant:", "", 1)
    if ":" in answer:
        answer = answer.split(":", 1)[1].strip()
    return answer


def default_question_template(question: str = None, apply_template=False, tokenizer=None) -> str:
    """
    Generates a default question template for a chatbot.

    Parameters:
    question (str): The question to be included in the template.

    Returns:
    list: A list of messages representing the template, with a system message and a user message.
    """
    formatted_question = f"{question}"
    if apply_template:
        messages = [
            # {"role": "system", "content": "You are a chatbot who can answer the question"},
            {"role": "user", "content": formatted_question},
        ]
        formatted_question = tokenizer.apply_chat_template(messages, tokenize=False)
    return formatted_question


def default_answer_template(answer: str, apply_template=False, tokenizer=None) -> str:
    """
    Generates a default answer template for a chatbot.

    Parameters:
    answer (str): The answer for which the template is generated.

    Returns:
    list: A list of messages representing the default answer template. Each message is a dictionary with 'role' and 'content' keys.
    """
    formatted_answer = f"What follows the colon is an answer from a query; can you guess what the query is？:\"{answer}\""
    if apply_template:
        messages = [
            # {
            #     "role": "system",
            #     "content": "You are a chatbot capable of guessing questions based on answers.",
            # },
            {"role": "user", "content": formatted_answer},
        ]
        formatted_answer = tokenizer.apply_chat_template(messages, tokenize=False)
    return formatted_answer


class Process:
    """
    a class to save the optional function in the evaluation,
    if not specified, we will create a default "Process"
    """

    def __init__(
        self,
        question_extract: Callable = default_question_extractor,
        answer_extract: Callable = default_answer_extractor,
        question_template: Callable = default_question_template,
        answer_template: Callable = default_answer_template,
        apply_template = False,
        tokenizer=None,
    ) -> None:
        self.question_extract = question_extract
        self.answer_extract = answer_extract
        self.question_template = question_template
        self.answer_template = answer_template
        self.apply_template = apply_template
        self.tokenizer = tokenizer

    def batch_question_extract(self, question_list: list[str],answer_list:list[str]) -> list[str]:
        """
        Extracts questions from a list of texts.

        Args:
            text_list (list of str): A list of text strings from which questions will be extracted.

        Returns:
            list of str: A list of extracted questions.
        """
        return [self.question_extract(question,answer) for question,answer in zip(question_list,answer_list)]

    def batch_answer_extract(self, answer_list: list[str],question_list: list[str]) -> list[str]:
        """
        Extracts answers from a batch of texts.

        Args:
            text_list (list of str): A list of text strings from which answers need to be extracted.

        Returns:
            list: A list of extracted answers corresponding to each text in the input list.
        """
        return [self.answer_extract(answer,question) for answer,question in zip(answer_list,question_list)]

    def batch_question_template(self, text_list) -> list[str]:
        """
        Applies the question_template method to each element in the provided list of texts.

        Args:
            text_list (list of str): A list of text strings to be processed.

        Returns:
            list of str: A list of processed text strings, where each string is the result of
                        applying the question_template method to the corresponding input string.
        """
        return [self.question_template(text,self.apply_template,self.tokenizer) for text in text_list]

    def batch_answer_template(self, text_list) -> list[str]:
        """
        Processes a list of texts using the answer_template method.

        Args:
            text_list (list of str): A list of text strings to be processed.

        Returns:
            list of str: A list of processed text strings.
        """
        return [self.answer_template(text,self.apply_template,self.tokenizer) for text in text_list]
