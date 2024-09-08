"""
the default process
"""

import re
from typing import Callable


def default_question_extractor(example, loop) -> str:
    """
    Extracts the default question from a given string.

    Parameters:
    q (str): The input string containing the question.

    Returns:
    str: The extracted default question.

    """
    answer = example[f"a{loop}"]
    question = example[f"q{loop+1}_output"]
    question = question.replace(answer, "", 1)
    split_text = question.split("The question is most likely")
    if len(split_text) > 1:
        question = split_text[-1].strip()
    example[f'q{loop+1}']=question
    return example


def default_answer_extractor(example, loop):
    answer = example[f"a{loop}_output"]
    question = example[f"q{loop}"]
    answer = answer.replace(question, "", 1)
    answer = answer.replace("Assistant:", "", 1)
    answer = answer.replace("assistant:", "", 1)
    if ":" in answer:
        answer = answer.split(":", 1)[1].strip()
    example[f"a{loop}"] = answer
    return example


def default_question_template(example, loop):
    example[f'q{loop}_prompt'] = example[f'q{loop}']
    return example


def default_answer_template(example, loop):
    answer = example[f"a{loop}"]
    answer = f"The following text comes from a response to a conversation, which most likely asks the following question?(Please reply in this format:The question is most likely......)\n\n{answer}"
    example[f"a{loop}_prompt"] = answer
    return example


def default_template(input:str)->list:
    message = [
        {
            "role": "system",
            "content": "You're a Q&A bot.",
        },
        {"role": "user", "content": input},
    ]
    return message


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
        chat_template:Callable = default_template
    ) -> None:
        self.question_extract = question_extract
        self.answer_extract = answer_extract
        self.question_template = question_template
        self.answer_template = answer_template
        self.chat_template = chat_template

    # def batch_question_extract(self, example,loop):
    #     """
    #     Extracts questions from a list of texts.

    #     Args:
    #         text_list (list of str): A list of text strings from which questions will be extracted.

    #     Returns:
    #         list of str: A list of extracted questions.
    #     """
    #     return [self.question_extract(question,answer) for question,answer in zip(question_list,answer_list)]

    # def batch_answer_extract(self, answer_list: list[str],question_list: list[str]) -> list[str]:
    #     """
    #     Extracts answers from a batch of texts.

    #     Args:
    #         text_list (list of str): A list of text strings from which answers need to be extracted.

    #     Returns:
    #         list: A list of extracted answers corresponding to each text in the input list.
    #     """
    #     return [self.answer_extract(answer,question) for answer,question in zip(answer_list,question_list)]

    # def batch_question_template(self, text_list) -> list[str]:
    #     """
    #     Applies the question_template method to each element in the provided list of texts.

    #     Args:
    #         text_list (list of str): A list of text strings to be processed.

    #     Returns:
    #         list of str: A list of processed text strings, where each string is the result of
    #                     applying the question_template method to the corresponding input string.
    #     """
    #     return [self.question_template(text) for text in text_list]

    # def batch_answer_template(self, text_list) -> list[str]:
    #     """
    #     Processes a list of texts using the answer_template method.

    #     Args:
    #         text_list (list of str): A list of text strings to be processed.

    #     Returns:
    #         list of str: A list of processed text strings.
    #     """
    #     return [self.answer_template(text) for text in text_list]
