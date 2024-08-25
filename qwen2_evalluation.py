
import pymongo
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
# from datasets import load_dataset
import evaluate
from tqdm import tqdm
import torch
import os
import random
import pandas as pd

# Set HTTP proxy
os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'
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
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

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


class Evaluation:
    def __init__(self, model,tokenizer,model_name, evaluation_data, language,device,backup_db):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.rouge = evaluate.load('rouge')
        self.model =model
        self.tokenizer = tokenizer
        self.database = backup_db
        self.evaluation_data = evaluation_data
        self.quesitons = None
        self.answers = None
   
    
    def recall_qa(self,ask,q0,loop,language):
        q_list= []
        a_list=[]
        q_list.append(q0)
        if language=='zh':
            prompt = '这个答案的问题最可能是什么:'
        elif language== 'en':
            prompt = 'What is the most likely question for this answer:'
        for i in range(loop):
            a = ask(self.model,self.tokenizer,self.device,q_list[i])
            a_list.append(a)
            q_next = model.chat(tokenizer, prompt+a, history=[])[0]
            q_list.append(q_next)
        return q_list, a_list


    def recall_qas(self,ask,question0_list,loop,language):
            question_records =[]
            answer_records =[]
            for i in tqdm(range(len(question0_list))):
                questions, answers = self.recall_qa(ask,question0_list[i], loop,language)
                question_records.append(questions)
                answer_records.append(answers)
            return question_records, answer_records
            
    def evalutaion(self,ask,loop):
        self.quesitons,self.answers = self.recall_qas(ask,self.evaluation_data,loop,self.language)
        return self.quesitons,self.answers

    def get_score(self):
        loop = len(self.answers[0])
        for i in range(loop):
            predictions = [answer[0] for answer in self.answers]
            references = [answer[i] for answer in self.answers]
            results = self.rouge.compute(predictions=predictions,references=references)
            print(f"loop{i}:{self.answers}")
        for i in range(1,loop):
            predictions = [answer[i-1] for answer in self.answers]
            references = [answer[i] for answer in self.answers]
            results = self.rouge.compute(predictions=predictions,references=references)
            print(f"loop{i}:{results}")

    def write2db(self):
        for i in range(len(self.answers)):
            rerord = {'model':self.model_name,'language':self.language,'question':self.quesitons[i],'answer':self.answers[i],'loop':loop}
            self.database.insert_one(rerord)
            

# device = "cuda" # the device to load the model onto
# model_checkpoint = "/public/model/Qwen2-7B"
# # Now you do not need to add "trust_remote_code=True"
device = 'cuda'
rouge = evaluate.load('rouge')
model_checkpoint = "/public/model/chatglm3-6b"
# ds = load_dataset("wanghw/human-ai-comparison",revision='d5d1b67')
# q0 = ds['train'].shuffle().select(range(10))['question']
loop = 2
document_count = 5
evaluation_data =pd.read_csv('q0.csv')['0'].tolist()[:document_count]
# q0 = random.sample(q0, document_count)
# evalution_data = [q['question'] for q in q0]
# print(q0[:2])
language = 'en'
writed_database = pymongo.MongoClient('10.48.48.7', 27017)['llm_evaluation']['test']
model = AutoModel.from_pretrained(model_checkpoint,trust_remote_code=True).half().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,trust_remote_code=True)
evalutaion = Evaluation(model,tokenizer,'chatglm-6b',evaluation_data,language,device,writed_database)
evalutaion.evalutaion(chatglm3_ask,loop)
evalutaion.get_score()