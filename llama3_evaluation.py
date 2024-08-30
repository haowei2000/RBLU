import pymongo
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
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
    rouge = evaluate.load("rouge")
    device = "auto"
    loop = 5
    document_count = 100
    model_name = 'llama3.1-8b'
    language = "en"
    model_id = "/public/model/Meta-Llama-3.1-8B/"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto").half()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    for field in ['code','finance','medical','law']:
        evaluation_data = pd.read_csv(f"data/data_{field}.csv")["question"].tolist()[:document_count]
        writed_database = pymongo.MongoClient("10.48.48.7", 27017)["llm_evaluation"][f"{model_name}_{field}"]
        evalutaion = Evaluation(
            model=model,
            tokenizer=tokenizer,
            metric=rouge,
            model_name=model_name,
            evaluation_data=evaluation_data,
            language=language,
            device=device,
            backup_db=writed_database,
            loop=loop,
            task=field,
            q_extractor=None,
            a_extractor=None
        )
        evalutaion.evalutate(llama_ask)
        # evalutaion.write2db()
        print(evalutaion.questions)
        print(evalutaion.answers)
        evalutaion.get_score('answer')
        evalutaion.get_score('question')
        # evalutaion.write_scores_to_csv()
    close_proxy()



if __name__ == "__main__":
    main()