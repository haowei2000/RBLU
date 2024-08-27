import os


def set_proxy():
    os.environ["HTTP_PROXY"] = "172.17.0.1:10809"
    os.environ["HTTPS_PROXY"] = "172.17.0.1:10809"
    print("HTTP_PROXY:", os.environ["HTTP_PROXY"])
    
def close_proxy():
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    print("HTTP_PROXY:", os.environ.get("HTTP_PROXY"))