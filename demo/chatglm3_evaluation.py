
import pymongo
from transformers import AutoTokenizer, AutoModelForCausalLM
# from datasets import load_dataset
import evaluate
import pandas as pd
from evaluation import Evaluation
from proxy import close_proxy, set_proxy

# Instead of using model.chat(), we directly use model.generate()
# But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
def chatglm3_ask(model,tokenizer,device,question)->str:
    return model.chat(tokenizer, question, history=[])[0]


def main():
    set_proxy()
    rouge = evaluate.load("rouge")
    device = "auto"
    loop = 10
    document_count = 100
    model_name = 'chatglm3-6b'
    language = "en"
    model_id = "/public/model/chatglm3-6b/"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",trust_remote_code=True).half()
    tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    for field in ['code','finance','medical','law']:
        evaluation_data = pd.read_csv(f"data/data_{field}.csv")["question"].tolist()[:document_count]
        writed_database = pymongo.MongoClient("10.48.48.7", 27017)["llm_evaluation"][f"{model_name}_{field}"]
        evalutaion = Evaluation(
            model=model,
            tokenizer=tokenizer,
            metric=rouge,
            model_name=model_name,
            original_questions=evaluation_data,
            language=language,
            device=device,
            backup_db=writed_database,
            loop=loop,
            task=field,
            q_extractor=None,
            a_extractor=None
        )
        evalutaion.evaluate(chatglm3_ask)
        evalutaion.write_qa2db()
        # print(evalutaion.questions)
        # print(evalutaion.answers)
        evalutaion.get_score('answer')
        evalutaion.get_score('question')
        evalutaion.write_scores_to_csv()
    close_proxy()


if __name__ == "__main__":
    main()