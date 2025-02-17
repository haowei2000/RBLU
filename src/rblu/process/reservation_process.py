"""the default processing functions for the reservation evaluation"""

from collections.abc import Callable
from dataclasses import dataclass


def extract_repharse(
    example: dict, loop: int, new_column: str, split_text: str
) -> dict:
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
    example[new_column] = prompt_text + example[f"q{loop}"]
    return example


def prompt_ask(
    example: dict, loop: int, new_column: str, prompt_text: str = ""
) -> dict:
    example[f"q{loop}_prompt2ask"] = f"{prompt_text}{example[f'q{loop}']}"
    return example


def extract_en_rephrase(example: dict, loop: int, new_column: str) -> dict:
    return extract_repharse(
        example,
        loop,
        new_column,
        "rephrase",
    )


def extract_zh_rephrase(example: dict, loop: int, new_column: str) -> dict:
    return extract_repharse(example, loop, new_column, "新的问题是")


def prompt_en_rephrase(example: dict, loop: int, new_column: str) -> dict:
    prompt_text = "Please express the following question in a different way:"
    return prompt_rephrase(example, loop, new_column, prompt_text)


def prompt_zh_rephrase(example: dict, loop: int, new_column: str) -> dict:
    prompt_text = "请将下面的问题换一个表达方式:"
    return prompt_rephrase(example, loop, new_column, prompt_text)


@dataclass
class ReservationProcess:
    """
    a class to save the optional function in the reservation,
    if not specified, we will create a default "Process"
    """

    prompt_rephrase: Callable[[dict, int, str], dict]
    extract_rephrase: Callable[[dict, int, str], dict]
    ask: Callable[[dict, int, str], dict]
    ask_extract: Callable[[dict, int, str], dict]


def get_reservation_process(language: str) -> ReservationProcess:
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
