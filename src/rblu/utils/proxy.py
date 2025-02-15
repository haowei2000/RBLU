"""set the system proxy on linux to use the transformers pickle file"""

import logging
import os


def set_proxy():
    """
    The function `set_proxy` sets HTTP and HTTPS proxy environment variables to
    "172.17.0.1:10810".
    """
    os.environ["HTTP_PROXY"] = "172.17.0.1:10810"
    os.environ["HTTPS_PROXY"] = "172.17.0.1:10810"
    os.environ["http_proxy"] = "172.17.0.1:10810"
    os.environ["https_proxy"] = "172.17.0.1:10810"


def close_proxy():
    """
    The function `close_proxy` removes HTTP_PROXY and HTTPS_PROXY environment
    variables and prints the value of HTTP_PROXY.
    """
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
