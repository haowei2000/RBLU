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
    ) -> None:
        self.question_extract = question_extract
        self.answer_extract = answer_extract
        self.question_template = question_template
        self.answer_template = answer_template
