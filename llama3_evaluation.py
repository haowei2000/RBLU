import pandas as pd
import transformers
import torch

model_id = "/public/model/Meta-Llama-3.1-8B"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)
question = "Will he be notified that there's a warrant for his arrest? Also how long until he's arrested?"
result=pipeline(question)
print(result)


import pymongo
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
# from datasets import load_dataset
import evaluate
from tqdm import tqdm
import torch
import os
import random
# Set HTTP proxy
os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'
# Set the proxies parameter when making HTTP requests
# Example:
# response = requests.get(url, proxies=proxies)

    

def recall_qa(q0,model,loop,language):
    q_list= []
    a_list=[]
    q_list.append(q0)
    if language=='zh':
        prompt = '这个答案的问题最可能是什么:'
    elif language== 'en':
        prompt = 'What is the most likely question for this answer:'
    for i in range(loop):
        # print(f'q{i}: {q_list[i]}')
        a = model.chat(tokenizer, q_list[i], history=[])[0]
        a_list.append(a)
        # print(f'a{i}: {a_list[i]}')
        q_next = model.chat(tokenizer, prompt+a, history=[])[0]
        q_list.append(q_next)
    return q_list, a_list


def recall_qas(question0_list,model,loop,language):
    question_records =[]
    answer_records =[]
    for i in tqdm(range(len(question0_list))):
        questions, answers = recall_qa(question0_list[i], model, loop,language)
        question_records.append(questions)
        answer_records.append(answers)
    return question_records, answer_records

device = 'cuda'
rouge = evaluate.load('rouge')
model_checkpoint = "/public/model/chatglm3-6b"
# ds = load_dataset("wanghw/human-ai-comparison",revision='d5d1b67')
# q0 = ds['train'].shuffle().select(range(10))['question']
loop = 10
document_count = 100
ds = pymongo.MongoClient('10.48.48.7', 27017)['QA']['backup_collection']
q0 =list(ds.find({'language':'en'},{'_id':0,'question':1}))
q0 = random.sample(q0, document_count)
q0 = [q['question'] for q in q0]
# print(q0[:2])
language = 'en'
writed_database = pymongo.MongoClient('10.48.48.7', 27017)['llm_evaluation']['test']
model = AutoModel.from_pretrained(model_checkpoint,trust_remote_code=True).half().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,trust_remote_code=True)
questions, answers = recall_qas(q0,model,loop,language)
for i in range(len(questions)):
    # print(f'question: {questions[i]}')
    # print(f'answer: {answers[i]}')
    # print('----------------------')
    rerord = {'model':'chatglm-6b','language':language,'question':questions[i],'answer':answers[i],'loop':loop}
    writed_database.insert_one(rerord)
print("score for the oringinal question:")
for i in range(loop):
    predictions = [answer[0] for answer in answers]
    references = [answer[i] for answer in answers]
    # predictions = [" ".join(pred.replace(" ", "")) for pred in predictions]
    # references = [" ".join(label.replace(" ", "")) for label in references]
    results = rouge.compute(predictions=predictions,references=references)
    print(f"loop{i}:{results}")
print("score for the last question:")
for i in range(1,loop):
    predictions = [answer[i-1] for answer in answers]
    references = [answer[i] for answer in answers]
    # predictions = [" ".join(pred.replace(" ", "")) for pred in predictions]
    # references = [" ".join(label.replace(" ", "")) for label in references]
    results = rouge.compute(predictions=predictions,references=references)
    print(f"loop{i}:{results}")