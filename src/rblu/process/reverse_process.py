"""the default processing functions for the reverse evaluation"""

from collections.abc import Callable
from dataclasses import dataclass


def extract_question(example: dict, loop: int, split_text: str) -> dict:
    answer = example[f"a{loop}"]
    question = example[f"q{loop + 1}_output"].replace(answer, "", 1)
    split = question.split(split_text)
    example[f"q{loop + 1}"] = split[-1].strip() if len(split) > 1 else question
    example[f"q{loop + 1}"] = example[f"q{loop + 1}"].strip("‘’”“：；\"'")
    return example


def extract_answer(example: dict, loop: int) -> dict:
    answer = example[f"a{loop}_output"].replace(example[f"q{loop}"], "", 1)
    answer = answer.replace("Assistant:", "", 1).replace("assistant:", "", 1)
    example[f"a{loop}"] = (
        answer.split(":", 1)[-1].strip() if ":" in answer else answer
    )
    return example


def prompt_question(example: dict, loop: int, prompt_text: str = "") -> dict:
    example[f"q{loop}_prompt"] = example[f"{prompt_text}{example[f'q{loop}']}"]
    return example


def prompt_answer(example: dict, loop: int, prompt_text: str = "") -> dict:
    example[f"a{loop}_prompt"] = f"{prompt_text}{example[f'a{loop}']}"
    return example


def extract_en_reverse_question(example: dict, loop: int) -> dict:
    return extract_question(example, loop, "The question is most likely")


def extract_zh_reverse_question(example: dict, loop: int) -> dict:
    return extract_question(example, loop, "该回答最可能的问题是")


def prompt_en_reverse_answer(example: dict, loop: int) -> dict:
    prompt_text = (
        "The following text comes from a response to a conversation,"
        "which most likely asks the following question?"
        "(Please reply in this format:The question is most likely......)"
    )
    return prompt_answer(example, loop, prompt_text)


def prompt_zh_reverse_answer(example: dict, loop: int) -> dict:
    prompt_text = (
        "下面的内容来自一段对话的回答，"
        "该回答最可能的问题是什么？"
        "(请用下面的格式回答:该回答最可能的问题是......)"
    )
    return prompt_answer(example, loop, prompt_text)


@dataclass
class ReverseProcess:
    """
    a class to save the optional function in the evaluation,
    if not specified, we will create a default "Process"
    """

    question_extract: Callable
    answer_extract: Callable
    question_prompt: Callable
    answer_prompt: Callable


def get_reverse_process(language: str) -> ReverseProcess:
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
