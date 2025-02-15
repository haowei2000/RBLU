import os


def parse_api(api_string: str) -> dict:
    """
    Parse the API string to get the url, model_name and key.
    Args:
        api_string (str): The string containing the url, model_name and key.
    """
    data_lists = api_string.split("--")
    if data_lists[-1] == "null":
        data_lists[-1] = os.getenv("CHATAPI_KEY", "default_api_key")
    return {
        "url": data_lists[1],
        "model_name": data_lists[2],
        "key": data_lists[3],
    }
