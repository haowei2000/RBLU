"""
The file contains the default functions to extract the question.
"""

# regular expression to extract string
import re


# it provides a default function to extract the question from a raw string
def default_q_extractor(q: str) -> str:
    """
    Extracts the default question from a given string.

    Parameters:
    q (str): The input string containing the question.

    Returns:
    str: The extracted default question.

    """
    match = re.search(r":\s*(.*)", q)
    if match:
        return match.group(1)
    else:
        return q


# it provides a default function to extract the question from a raw string
def default_a_extractor(a: str) -> str:
    """
    Extracts the default value of 'a'.

    Parameters:
        a (str): The input string.

    Returns:
        str: The extracted value of 'a'.
    """
    return a


# End-of-file (EOF)
