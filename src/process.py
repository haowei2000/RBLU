"""the default process"""

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
    """
    Extracts and cleans the answer from the given example based on the loop index.

    This function retrieves the answer and question from the example dictionary using the loop index.
    It then removes the question and any leading "Assistant:" or "assistant:" prefixes from the answer.
    If there is a colon (":") in the answer, it splits the answer at the first colon and trims any leading/trailing whitespace.
    The cleaned answer is then stored back in the example dictionary with the key `f"a{loop}"`.

    Args:
        example (dict): A dictionary containing the example data with keys formatted as `f"a{loop}_output"` and `f"q{loop}"`.
        loop (int): The loop index used to access the specific question and answer in the example dictionary.

    Returns:
        dict: The updated example dictionary with the cleaned answer stored under the key `f"a{loop}"`.
    """
    answer = example[f"a{loop}_output"]
    question = example[f"q{loop}"]
    answer = answer.replace(question, "", 1)
    answer = answer.replace("Assistant:", "", 1)
    answer = answer.replace("assistant:", "", 1)
    if ":" in answer:
        answer = answer.split(":", 1)[1].strip()
    example[f"a{loop}"] = answer
    return example


def default_question_prompt(example, loop):
    """
    Updates the given example dictionary by copying the value of the key 'q{loop}' 
    to a new key 'q{loop}_prompt'.

    Args:
        example (dict): The dictionary containing the example data.
        loop (int): The loop index used to generate the key names.

    Returns:
        dict: The updated example dictionary with the new 'q{loop}_prompt' key.
    """
    example[f'q{loop}_prompt'] = example[f'q{loop}']
    return example


def default_answer_prompt(example, loop):
    """
    Generates a prompt for a given example and loop index.

    This function takes an example dictionary and a loop index, retrieves the answer
    corresponding to the loop index, formats it into a specific prompt template, and
    adds the formatted prompt back into the example dictionary with a new key.

    Args:
        example (dict): A dictionary containing the example data.
        loop (int): The loop index to retrieve the answer from the example.

    Returns:
        dict: The updated example dictionary with the new prompt added.
    """
    answer = example[f"a{loop}"]
    answer = f"The following text comes from a response to a conversation, which most likely asks the following question?(Please reply in this format:The question is most likely......)\n\n{answer}"
    example[f"a{loop}_prompt"] = answer
    return example


def apply_default_template(user_input:str)->list:
    """
    Generates a default message template for a Q&A bot.

    Args:
        user_input (str): The user's input message.

    Returns:
        list: A list of dictionaries representing the message template, 
            where the first dictionary sets the role to "system" with a 
            content indicating the bot's role, and the second dictionary 
            contains the user's input.
    """
    message = [
        {
            "role": "system",
            "content": "You're a Q&A bot.",
        },
        {"role": "user", "content": user_input},
    ]
    return message

def apply_gemma_template(user_input:str)->list:
    """
    Generates a gemma message template.

    Args:
        user_input (str): The user's input message.

    Returns:
        list: A list of dictionaries representing the message template, 
            where the first dictionary sets the role to "system" with a 
            content indicating the bot's role, and the second dictionary 
            contains the user's input.
    """
    message = [
        {"role": "user", "content": user_input},
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
        question_prompt: Callable = default_question_prompt,
        answer_prompt: Callable = default_answer_prompt,
    ) -> None:
        self.question_extract = question_extract
        self.answer_extract = answer_extract
        self.question_template = question_prompt
        self.answer_template = answer_prompt
