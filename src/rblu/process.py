"""the default process"""

from typing import Callable, Dict
from dataclasses import dataclass

def extract_question(example: Dict, loop: int, split_text: str) -> Dict:
    answer = example[f"a{loop}"]
    question = example[f"q{loop + 1}_output"].replace(answer, "", 1)
    split = question.split(split_text)
    example[f"q{loop + 1}"] = split[-1].strip() if len(split) > 1 else question
    example[f"q{loop + 1}"] = example[f"q{loop + 1}"].strip("‘’”“：；\"\'")
    return example


def default_question_extract(example: Dict, loop: int) -> Dict:
    return extract_question(example, loop, "The question is most likely")


def zh_question_extract(example: Dict, loop: int) -> Dict:
    return extract_question(example, loop, "该回答最可能的问题是")


def default_answer_extract(example: Dict, loop: int) -> Dict:
    answer = example[f"a{loop}_output"].replace(example[f"q{loop}"], "", 1)
    answer = answer.replace("Assistant:", "", 1).replace("assistant:", "", 1)
    example[f"a{loop}"] = (
        answer.split(":", 1)[-1].strip() if ":" in answer else answer
    )
    return example


def default_question_prompt(example: Dict, loop: int) -> Dict:
    example[f"q{loop}_prompt"] = example[f"q{loop}"]
    return example


def generate_answer_prompt(example: Dict, loop: int, prompt_text: str) -> Dict:
    answer = example[f"a{loop}"]
    example[f"a{loop}_prompt"] = f"{prompt_text}\n\n{answer}"
    return example


def default_answer_prompt(example: Dict, loop: int) -> Dict:
    prompt_text = (
        "The following text comes from a response to a conversation,"
        "which most likely asks the following question?"
        "(Please reply in this format:The question is most likely......)"
    )
    return generate_answer_prompt(example, loop, prompt_text)


def zh_answer_prompt(example: Dict, loop: int) -> Dict:
    prompt_text = (
        "下面的内容来自一段对话的回答，"
        "该回答最可能的问题是什么？"
        "(请用下面的格式回答:该回答最可能的问题是......)"
    )
    return generate_answer_prompt(example, loop, prompt_text)


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

    question_extract: Callable = default_question_extract
    answer_extract: Callable = default_answer_extract
    question_prompt: Callable = default_question_prompt
    answer_prompt: Callable = default_answer_prompt


def get_process(language):
    if language == "zh":
        return Process(
            question_extract=zh_question_extract,
            answer_extract=default_answer_extract,
            question_prompt=default_question_prompt,
            answer_prompt=zh_answer_prompt,
        )
    return Process(
        question_extract=default_question_extract,
        answer_extract=default_answer_extract,
        question_prompt=default_question_prompt,
        answer_prompt=default_answer_prompt,
    )
