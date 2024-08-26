
import pymongo
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
# from datasets import load_dataset
import evaluate
from tqdm import tqdm
import torch
import os
import random
import pandas as pd

from evaluation import Evaluation

# Set HTTP proxy
os.environ['HTTP_PROXY'] = '172.17.0.1:10809'
os.environ['HTTPS_PROXY'] = '172.17.0.1:10809'
print('HTTP_PROXY:', os.environ['HTTP_PROXY'])
# Set the proxies parameter when making HTTP requests
# Example:
# response = requests.get(url, proxies=proxies)
from transformers import AutoModelForCausalLM, AutoTokenizer

# Instead of using model.chat(), we directly use model.generate()
# But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
def chatglm3_ask(model,tokenzier,device,question)->str:
    return model.chat(tokenizer, question, history=[])[0]

def qwen_ask(model,tokenizer,deivce,question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").cuda()

    # Directly use generate() and tokenizer.decode() to get the output.
    # Use `max_new_tokens` to control the maximum output length.
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

            


device = 'cuda'
rouge = evaluate.load('rouge')
model_checkpoint = "/public/model/chatglm3-6b/"
loop = 3
document_count = 5
evaluation_data =pd.read_csv('q0.csv')['0'].tolist()[:document_count]
# q0 = random.sample(q0, document_count)
# evalution_data = [q['question'] for q in q0]
# print(q0[:2])
language = 'en'
writed_database = pymongo.MongoClient('10.48.48.7', 27017)['llm_evaluation']['test']
model = AutoModel.from_pretrained(model_checkpoint,trust_remote_code=True).half().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,device_map='auto',trust_remote_code=True)
evalutaion = Evaluation(model,tokenizer,'chatglm-6b',evaluation_data,language,device,writed_database)
evalutaion.evaluate(chatglm3_ask,loop)
evalutaion.get_score()
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
print('HTTP_PROXY:', os.environ.get('HTTP_PROXY'))