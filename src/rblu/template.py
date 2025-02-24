"""Apply a template to the user input."""


def apply_template(user_input: str, system_content: str) -> list:
    """
    Creates a list of dictionaries representing a conversation template.

    Args:
        user_input (str): The content provided by the user.
        system_content (str): The content provided by the system.

    Returns:
        list: A list of dictionaries with roles and their corresponding content.
    """
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input},
    ]


def apply_default_template(user_input: str) -> list:
    """
    Applies a default template to the given user input.

    This function takes a user input string and applies a default template
    to it, which is "You're a Q&A bot.".

    Args:
        user_input (str): The input string provided by the user.

    Returns:
        list: The result of applying the template to the user input.
    """
    return apply_template(user_input, "You're a Q&A bot.")


def apply_default_zh_template(user_input: str) -> list:
    """
    Applies a default Chinese template to the user input.

    This function takes a user input string and applies a default template
    that sets the context as a Chinese chatbot.

    Args:
        user_input (str): The input string from the user.

    Returns:
        list: The result of applying the template to the user input.
    """
    return apply_template(user_input, "你是一个聊天问答机器人")


def apply_gemma_template(user_input: str) -> list:
    """
    Applies the Gemma template to the given user input.

    Args:
        user_input (str): The input string provided by the user.

    Returns:
        list: A list containing a dictionary with the role set to "user" and the content set to the user input.
    """
    return [{"role": "user", "content": user_input}]
