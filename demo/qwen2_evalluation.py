import pymongo
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
import pandas as pd
from evaluation import Evaluation
from transformers import AutoModelForCausalLM, AutoTokenizer
from proxy import close_proxy, set_proxy

def qwen_ask(model, tokenizer, device, question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

    # Directly use generate() and tokenizer.decode() to get the output.
    # Use `max_new_tokens` to control the maximum output length.
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# ds = load_dataset("wanghw/human-ai-comparison",revision='d5d1b67')
# q0 = ds['train'].shuffle().select(range(10))['question']
# q0 = random.sample(q0, document_count)
# evalution_data = [q['question'] for q in q0]
# print(q0[:2])
def main():
    set_proxy()
    device = "auto"
    rouge = evaluate.load("rouge")
    loop = 10
    document_count = 100
    model_name = 'qwen2-7b'
    language = "en"
    model_id = "/public/model/Qwen2-7B/"
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
        evalutaion.evaluate(qwen_ask)
        evalutaion.write_qa2db()
        # print(evalutaion.questions)
        # print(evalutaion.answers)
        evalutaion.get_score('answer')
        evalutaion.get_score('question')
        evalutaion.write_scores_to_csv()
    close_proxy()
    

if __name__ == "__main__":
    main()