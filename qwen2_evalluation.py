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
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

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
    model_checkpoint = "/public/model/Qwen2-7B/"
    loop = 10
    document_count = 100
    evaluation_data = pd.read_csv("q0.csv")["0"].tolist()[:document_count]
    language = "en"
    writed_database = pymongo.MongoClient("10.48.48.7", 27017)["llm_evaluation"]["test"]
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # model = AutoModel.from_pretrained(model_checkpoint,trust_remote_code=True).half().to(device)
    # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,trust_remote_code=True)
    evalutaion = Evaluation(
        model=model,
        tokenizer=tokenizer,
        metric=rouge,
        model_name="qwem2-7b",
        evaluation_data=evaluation_data,
        language=language,
        device='cuda',
        backup_db=writed_database,
        loop=loop,
    )
    evalutaion.evalutate(qwen_ask)
    evalutaion.get_score()
    close_proxy()
    

if __name__ == "__main__":
    main()