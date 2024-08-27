import pymongo
import transformers
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import evaluate
import torch
import pandas as pd
from evaluation import Evaluation
from proxy import close_proxy, set_proxy



def llama_ask(model, tokenizer, deivce, question):
    model_inputs = tokenizer([question], return_tensors="pt").to('cuda')
    generated_ids = model.generate(**model_inputs,max_new_tokens=512).to('cuda')
    result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return result





def main():
    set_proxy()
    device = "auto"
    rouge = evaluate.load("rouge")
    loop = 10
    document_count = 100
    evaluation_data = pd.read_csv("q0.csv")["0"].tolist()[:document_count]
    language = "en"
    writed_database = pymongo.MongoClient("10.48.48.7", 27017)["llm_evaluation"]["llama3.1-8b"]
    model_id = "/public/model/Meta-Llama-3.1-8B/"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto").half()
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
    evalutaion.get_score('answer')
    evalutaion.get_score('question')
    close_proxy()



if __name__ == "__main__":
    main()