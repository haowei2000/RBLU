
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