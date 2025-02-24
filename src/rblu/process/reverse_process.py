"""The default processing functions for the reverse evaluation"""

from collections.abc import Callable
from dataclasses import dataclass


def extract_question(example: dict, loop: int, split_text: str) -> dict:
    """
    Extracts and processes a question from the given example dictionary.

    Args:
        example (dict): A dictionary containing question and answer pairs. loop
        (int): The current loop index used to access specific question and
        answer keys. split_text (str): The text used to split the question for
        further processing.

    Returns:
        dict: The updated example dictionary with the processed question.
    """
    answer = example[f"a{loop}"]
    question = example[f"q{loop + 1}_output"].replace(answer, "", 1)
    split = question.split(split_text)
    example[f"q{loop + 1}"] = split[-1].strip() if len(split) > 1 else question
    example[f"q{loop + 1}"] = example[f"q{loop + 1}"].strip("‘’”“：；\"'")
    return example


def extract_answer(example: dict, loop: int) -> dict:
    """
    Extracts and processes the answer from a given example dictionary.

    Args:
        example (dict): A dictionary containing question and answer pairs. loop
        (int): The index of the question-answer pair to process.

    Returns:
        dict: The updated example dictionary with the processed answer.
    """
    answer = example[f"a{loop}_output"].replace(example[f"q{loop}"], "", 1)
    answer = answer.replace("Assistant:", "", 1).replace("assistant:", "", 1)
    example[f"a{loop}"] = (
        answer.split(":", 1)[-1].strip() if ":" in answer else answer
    )
    return example


def prompt_question(example: dict, loop: int, prompt_text: str = "") -> dict:
    """
    Appends a prompt text to a specific question in the example dictionary.

    Args:
        example (dict): The dictionary containing the questions. loop (int): The
        index of the question to which the prompt text will be appended.
        prompt_text (str, optional): The text to prepend to the question.
        Defaults to an empty string.

    Returns:
        dict: The updated dictionary with the appended prompt text.
    """
    example[f"q{loop}_prompt"] = f"{prompt_text}{example[f'q{loop}']}"
    return example


def prompt_answer(example: dict, loop: int, prompt_text: str = "") -> dict:
    """
    Appends a prompt text to a specific key in the example dictionary.

    Args:
        example (dict): The dictionary containing the example data. loop (int):
        The loop index used to identify the specific key in the dictionary.
        prompt_text (str, optional): The text to prepend to the value of the
        specific key. Defaults to an empty string.

    Returns:
        dict: The updated dictionary with the modified key-value pair.
    """
    example[f"a{loop}_prompt"] = f"{prompt_text}{example[f'a{loop}']}"
    return example


def extract_en_reverse_question(example: dict, loop: int) -> dict:
    """
    Extracts an English reverse question from the given example.

    Args:
        example (dict): A dictionary containing the example data.
        loop (int): An integer representing the loop iteration.

    Returns:
        dict: A dictionary containing the extracted question.
    """
    return extract_question(example, loop, "The question is most likely")


def extract_zh_reverse_question(example: dict, loop: int) -> dict:
    """
    Extracts a reverse question in Chinese from the given example.

    Args:
        example (dict): The input example containing data to extract the
        question from. loop (int): An integer representing the loop iteration or
        index.

    Returns:
        dict: A dictionary containing the extracted reverse question.

    """
    return extract_question(example, loop, "该回答最可能的问题是")


def prompt_en_reverse_answer(example: dict, loop: int) -> dict:
    """
    Generates a prompt text for reverse engineering an answer to determine the most likely question.

    Args:
        example (dict): A dictionary containing the example data.
        loop (int): An integer representing the loop iteration.

    Returns:
        dict: A dictionary containing the generated prompt text.
    """
    prompt_text = (
        "The following text comes from a response to a conversation,"
        "which most likely asks the following question?"
        "(Please reply in this format:The question is most likely......)"
    )
    return prompt_answer(example, loop, prompt_text)


def prompt_zh_reverse_answer(example: dict, loop: int) -> dict:
    """
    Generates a prompt in Chinese to reverse-engineer the most likely question
    for a given answer in a conversation.

    Args:
        example (dict): A dictionary containing the example data.
        loop (int): An integer representing the loop iteration.

    Returns:
        dict: A dictionary containing the generated prompt and other relevant information.
    """
    prompt_text = (
        "下面的内容来自一段对话的回答，"
        "该回答最可能的问题是什么？"
        "(请用下面的格式回答:该回答最可能的问题是......)"
    )
    return prompt_answer(example, loop, prompt_text)


@dataclass
class ReverseProcess:
    """
    A class to save the optional function in the evaluation,
    if not specified, we will create a default "Process"
    """

    question_extract: Callable
    answer_extract: Callable
    question_prompt: Callable
    answer_prompt: Callable


def get_reverse_process(language: str) -> ReverseProcess:
    """
    Returns a ReverseProcess object configured for the specified language.

    Args:
        language (str): The language code for which the reverse process is
        needed.
                        Supported values are "zh" for Chinese and "en" for
                        English.

    Returns:
        ReverseProcess: An instance of ReverseProcess configured with the
        appropriate
                        extraction and prompting functions for the specified
                        language.

    Raises:
        ValueError: If the specified language is not supported.
    """
    match language:
        case "zh":
            reverse_process = ReverseProcess(
                question_extract=extract_zh_reverse_question,
                answer_extract=extract_answer,
                question_prompt=prompt_question,
                answer_prompt=prompt_zh_reverse_answer,
            )
        case "en":
            reverse_process = ReverseProcess(
                question_extract=extract_en_reverse_question,
                answer_extract=extract_answer,
                question_prompt=prompt_question,
                answer_prompt=prompt_en_reverse_answer,
            )
        case _:
            raise ValueError(f"The language {language} is not supported")
    return reverse_process
