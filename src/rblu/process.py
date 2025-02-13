"""the default process"""

from collections.abc import Callable
from dataclasses import dataclass


def extract_question_process(
    example: dict, loop: int, split_text: str
) -> dict:
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


def prompt_answer(example: dict, loop: int, prompt_text: str) -> dict:
    answer = example[f"a{loop}"]
    example[f"a{loop}_prompt"] = f"{prompt_text}\n\n{answer}"
    return example


def prompt_question(example: dict, loop: int) -> dict:
    example[f"q{loop}_prompt"] = example[f"q{loop}"]
    return example


def reverse_question_extract_en(example: dict, loop: int) -> dict:
    return extract_question_process(
        example, loop, "The question is most likely"
    )


def reverse_question_extract_zh(example: dict, loop: int) -> dict:
    return extract_question_process(example, loop, "该回答最可能的问题是")


# TODO add the function to extract the question from the reservation data
def reservation_question_extract_en(example: dict, loop: int) -> dict:
    pass


# TODO add the function to extract the question from the reservation data
def reservation_question_extract_zh(example: dict, loop: int) -> dict:
    pass


def reverse_prompt_en(example: dict, loop: int) -> dict:
    prompt_text = (
        "The following text comes from a response to a conversation,"
        "which most likely asks the following question?"
        "(Please reply in this format:The question is most likely......)"
    )
    return prompt_answer(example, loop, prompt_text)


def reverse_prompt_zh(example: dict, loop: int) -> dict:
    prompt_text = (
        "下面的内容来自一段对话的回答，"
        "该回答最可能的问题是什么？"
        "(请用下面的格式回答:该回答最可能的问题是......)"
    )
    return prompt_answer(example, loop, prompt_text)


def reservation_prompt_en(example: dict, loop: int) -> dict:
    prompt_text = "Please express the following question in a different way:\n"
    return prompt_answer(example, loop, prompt_text)


def reservation_prompt_zh(example: dict, loop: int) -> dict:
    prompt_text = "请将下面的问题换一种表达方式:\n"
    return prompt_answer(example, loop, prompt_text)


def apply_template(user_input: str, system_content: str) -> list:
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input},
    ]


def apply_default_template(user_input: str) -> list:
    return apply_template(user_input, "You're a Q&A bot.")


def apply_default_zh_template(user_input: str) -> list:
    return apply_template(user_input, "你是一个聊天问答机器人")


def apply_gemma_template(user_input: str) -> list:
    return [{"role": "user", "content": user_input}]


@dataclass
class Process:
    """
    a class to save the optional function in the evaluation,
    if not specified, we will create a default "Process"
    """

    question_extract: Callable = reverse_question_extract_en
    answer_extract: Callable = extract_answer
    question_prompt: Callable = prompt_question
    answer_prompt: Callable = reverse_prompt_en


def get_process(language: str, stage: str) -> Process:
    match {"language": language, "stage": stage}:
        case {"language": "zh", "stage": "reverse"}:
            process = Process(
                question_extract=reverse_question_extract_zh,
                answer_extract=extract_answer,
                question_prompt=prompt_question,
                answer_prompt=reverse_prompt_zh,
            )
        case {"language": "en", "stage": "reverse"}:
            process = Process(
                question_extract=reverse_question_extract_en,
                answer_extract=extract_answer,
                question_prompt=prompt_question,
                answer_prompt=reverse_prompt_en,
            )
        case {"language": "zh", "stage": "reservation"}:
            process = Process(
                question_extract=reservation_question_extract_zh,
                answer_extract=extract_answer,
                question_prompt=prompt_question,
                answer_prompt=reservation_prompt_zh,
            )
        case {"language": "en", "stage": "reservation"}:
            process = Process(
                question_extract=reservation_question_extract_en,
                answer_extract=extract_answer,
                question_prompt=prompt_question,
                answer_prompt=reservation_prompt_en,
            )
        case _:
            raise ValueError("The language or stage is not supported")
    return process
