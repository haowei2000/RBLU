import re


def default_q_extractor(q:str)->str:
    match = re.search(r":\s*(.*)", q)
    if match:
        return match.group(1)
    else:
        return q

def default_a_extractor(a:str)->str:
    return a