"""the default processing functions for the reservation evaluation"""

from collections.abc import Callable
from dataclasses import dataclass


def extract_repharse(
    example: dict, loop: int, new_column: str, split_text: str
) -> dict:
    """
    Extracts and rephrases a question from the given example dictionary based on
    the loop index and split text.

    Args:
        example (dict): The dictionary containing the question to be processed.
        loop (int): The current loop index used to identify the question key.
        new_column (str): The key under which the rephrased question will be
        stored in the dictionary. split_text (str): The text used to determine
        how to split and rephrase the question.

    Returns:
        dict: The updated dictionary with the rephrased question stored under
        the new column key.
    """
    question = example[f"q{loop + 1}_unextracted"]
    if ":" in question and split_text in question:
        text_list = question.split(":", 1)
        if split_text in text_list[0]:
            question = text_list[1]
        else:
            question = text_list[0] + text_list[1]
    example[new_column] = question.strip("‘’”“：；\"'")
    return example


def extract_answer(example: dict, loop: int, new_column: str) -> dict:
    """
    Extracts and processes an answer from the given example dictionary.

    Args:
        example (dict): The dictionary containing the example data. loop (int):
        The loop index used to access specific keys in the dictionary.
        new_column (str): The key under which the processed answer will be
        stored in the dictionary.

    Returns:
        dict: The updated dictionary with the processed answer stored under the
        new_column key.
    """
    answer = example[f"a{loop}_unextracted"].replace(
        example[f"q{loop}"], "", 1
    )
    answer = answer.replace("Assistant:", "", 1).replace("assistant:", "", 1)
    answer = answer.split(":", 1)[-1].strip() if ":" in answer[:20] else answer
    example[new_column] = answer
    return example


def prompt_rephrase(
    example: dict, loop: int, new_column: str, prompt_text: str = ""
) -> dict:
    """
    Adds a new key-value pair to the given dictionary by concatenating a prompt
    text with a specific key's value.

    Args:
        example (dict): The dictionary to be updated. loop (int): The loop index
        used to form the key name in the dictionary. new_column (str): The key
        name for the new entry in the dictionary. prompt_text (str, optional):
        The text to prepend to the value from the dictionary. Defaults to an
        empty string.

    Returns:
        dict: The updated dictionary with the new key-value pair.
    """
    example[new_column] = prompt_text + example[f"q{loop}"]
    return example


def prompt_ask(
    example: dict, loop: int, new_column: str, prompt_text: str = ""
) -> dict:
    """
    Adds a new key-value pair to the given dictionary by concatenating a prompt
    text with a specific value from the dictionary.

    Args:
        example (dict): The dictionary to be updated. loop (int): The loop index
        used to access a specific key in the dictionary. new_column (str): The
        key for the new entry in the dictionary. prompt_text (str, optional):
        The text to be concatenated with the dictionary value. Defaults to an
        empty string.

    Returns:
        dict: The updated dictionary with the new key-value pair.
    """
    example[new_column] = prompt_text + example[f"q{loop}"]
    return example


def extract_en_rephrase(example: dict, loop: int, new_column: str) -> dict:
    """
    Extracts and rephrases a given example dictionary.

    This function is a wrapper around the `extract_repharse` function,
    specifically for rephrasing tasks. It takes an example dictionary, a loop
    count, and a new column name, and returns a rephrased version of the
    example.

    Args:
        example (dict): The input example dictionary to be rephrased. loop
        (int): The number of times to loop through the rephrasing process.
        new_column (str): The name of the new column to store the rephrased
        text.

    Returns:
        dict: The rephrased example dictionary.
    """
    return extract_repharse(
        example,
        loop,
        new_column,
        "rephrase",
    )


def extract_zh_rephrase(example: dict, loop: int, new_column: str) -> dict:
    """
    Extracts and rephrases a Chinese question from the given example dictionary.

    Args:
        example (dict): The input dictionary containing the data to be
        processed. loop (int): The loop count or iteration number. new_column
        (str): The name of the new column to store the rephrased question.

    Returns:
        dict: The updated dictionary with the rephrased question added under the
        specified new column.
    """
    return extract_repharse(example, loop, new_column, "新的问题是")


def prompt_en_rephrase(example: dict, loop: int, new_column: str) -> dict:
    """
    Rephrases a given question in English.

    Args:
        example (dict): A dictionary containing the example data. loop (int):
        The number of times to attempt rephrasing. new_column (str): The name of
        the new column to store the rephrased question.

    Returns:
        dict: The updated dictionary with the rephrased question.
    """
    prompt_text = "Please express the following question in a different way:"
    return prompt_rephrase(example, loop, new_column, prompt_text)


def prompt_zh_rephrase(example: dict, loop: int, new_column: str) -> dict:
    """
    Rephrases a given example in Chinese using a specified prompt text.

    Args:
        example (dict): The input example to be rephrased.
        loop (int): The number of times to loop through the rephrasing process.
        new_column (str): The name of the new column to store the rephrased text.

    Returns:
        dict: The rephrased example with the new column added.
    """
    prompt_text = "请将下面的问题换一个表达方式:"
    return prompt_rephrase(example, loop, new_column, prompt_text)


@dataclass
class ReservationProcess:
    """
    A class to save the optional function in the reservation,
    if not specified, we will create a default "Process"
    """

    prompt_rephrase: Callable[[dict, int, str], dict]
    extract_rephrase: Callable[[dict, int, str], dict]
    ask: Callable[[dict, int, str], dict]
    ask_extract: Callable[[dict, int, str], dict]


def get_reservation_process(language: str) -> ReservationProcess:
    """
    Returns a ReservationProcess object configured for the specified language.

    Args:
        language (str): The language code for which the reservation process
        should be configured.
                        Supported values are "zh" for Chinese and "en" for
                        English.

    Returns:
        ReservationProcess: An instance of ReservationProcess configured with
        the appropriate
                            rephrase and extraction functions based on the
                            specified language.

    Raises:
        ValueError: If the specified language is not supported.
    """
    match language:
        case "zh":
            reservation_process = ReservationProcess(
                prompt_rephrase=prompt_zh_rephrase,
                extract_rephrase=extract_zh_rephrase,
                ask=prompt_ask,
                ask_extract=extract_answer,
            )
        case "en":
            reservation_process = ReservationProcess(
                prompt_rephrase=prompt_en_rephrase,
                extract_rephrase=extract_en_rephrase,
                ask=prompt_ask,
                ask_extract=extract_answer,
            )
        case _:
            raise ValueError(f"The language {language} is not supported")
    return reservation_process
