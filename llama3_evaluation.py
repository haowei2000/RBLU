import pymongo
import transformers
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import evaluate
import torch
import os
import pandas as pd
from evaluation import Evaluation

# Set HTTP proxy
# os.environ["HTTP_PROXY"] = "127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "127.0.0.1:7890"
# print("HTTP_PROXY:", os.environ["HTTP_PROXY"])


def llama_ask(model, tokenizer, deivce, question):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=deivce,
        max_new_tokens=512,
    )
    answer = pipeline(question)
    return answer[0]['generated_text']

def main():
    device = "auto"
    rouge = evaluate.load("rouge")
    loop = 3
    document_count = 5
    evaluation_data = pd.read_csv("q0.csv")["0"].tolist()[:document_count]
    language = "en"
    writed_database = pymongo.MongoClient("10.48.48.7", 27017)["llm_evaluation"]["test"]
    model_id = "/public/model/Meta-Llama-3.1-8B/"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    evalutaion = Evaluation(
        model=model,
        tokenizer=tokenizer,
        metric=rouge,
        model_name="llama3.1-8b",
        evaluation_data=evaluation_data,
        language=language,
        device=device,
        backup_db=writed_database,
        loop=loop,
    )
    evalutaion.evalutate(llama_ask)
    evalutaion.get_score()
    # os.environ.pop("HTTP_PROXY", None)
    # os.environ.pop("HTTPS_PROXY", None)
    # print("HTTP_PROXY:", os.environ.get("HTTP_PROXY"))


if __name__ == "__main__":
    main()