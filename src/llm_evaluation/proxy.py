"""
set the system proxy on linux to use the transformers pickle
"""

import os


def set_proxy():
    """
    The function `set_proxy` sets HTTP and HTTPS proxy environment variables to
    "172.17.0.1:10809".
    """
    os.environ["HTTP_PROXY"] = "172.17.0.1:10809"
    os.environ["HTTPS_PROXY"] = "172.17.0.1:10809"
    print("HTTP_PROXY:", os.environ["HTTP_PROXY"])


def close_proxy():
    """
    The function `close_proxy` removes HTTP_PROXY and HTTPS_PROXY environment
    variables and prints the value of HTTP_PROXY.
    """
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    print("HTTP_PROXY:", os.environ.get("HTTP_PROXY"))
