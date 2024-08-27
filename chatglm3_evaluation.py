
import pymongo
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
# from datasets import load_dataset
import evaluate
import pandas as pd
from evaluation import Evaluation
from proxy import close_proxy, set_proxy
from transformers import AutoModelForCausalLM, AutoTokenizer

# Instead of using model.chat(), we directly use model.generate()
# But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
def chatglm3_ask(model,tokenizer,device,question)->str:
    return model.chat(tokenizer, question, history=[])[0]


def main():
    set_proxy()
    device = "auto"
    rouge = evaluate.load("rouge")
    loop = 10
    document_count = 100
    evaluation_data = pd.read_csv("q0.csv")["0"].tolist()[:document_count]
    language = "en"
    writed_database = pymongo.MongoClient("10.48.48.7", 27017)["llm_evaluation"]["chatglm3-6b"]     
    model_checkpoint = "/public/model/chatglm3-6b/"
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map="auto",trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
    evalutaion = Evaluation(
        model=model,
        tokenizer=tokenizer,
        metric=rouge,
        model_name="chatglm3",
        evaluation_data=evaluation_data,
        language=language,
        device=device,
        backup_db=writed_database,
        loop=loop,
    )
    evalutaion.evalutate(chatglm3_ask)
    evalutaion.get_score()
    evalutaion.write_scores_to_csv()
    evalutaion.write2db()
    close_proxy()



if __name__ == "__main__":
    main()