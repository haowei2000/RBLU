"""
This module provides functions to translate names of languages, models, etc.
"""


def translate_language(language_code: str) -> str:
    """Translate a language code to its full name.

    This function takes a language code (e.g., "zh", "en") and returns its corresponding full name (e.g., "chinese", "english"). If the code is not found in the translation dictionary, it returns "unknown".

    Args:
        code: The language code to translate.

    Returns:
        The full name of the language, or "unknown" if the code is not found.
    """
    translation_dict = {"zh": "chinese", "en": "english"}
    return translation_dict.get(language_code, "unknown")


def translate_model(model_name: str, with_refer=False) -> str:
    """Translate a model name to its full name.

    This function takes a model name (e.g., "llama", "glm") and returns its corresponding full name (e.g., "LLAMA3.1", "GLM4"). If the model name is not found in the translation dictionary, it returns "unknown". It also handles suffixes for referential models.

    Args:
        model_name: The model name to translate.
        with_refer: Whether the model name includes a referential suffix.

    Returns:
        The full name of the model, or "unknown" if the model name is not found.
    """
    suffix = ""
    translation_dict = {
        "llama": "LLAMA3.1",
        "glm": "GLM4",
        "qwen": "Qwen2",
        "gpt-4o-mini": "gpt",
    }
    suffix_dict: dict[str, str] = {"n-1": "Previous", "0": "Original"}
    if with_refer:
        model_name, suffix = model_name.split(sep=" ")
    supper_model_name = translation_dict.get(model_name, "unknown")
    supper_suffix = suffix_dict.get(suffix, "unknown")
    return (
        f"{supper_model_name}-{supper_suffix}"
        if with_refer
        else supper_model_name
    )
