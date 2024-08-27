import evaluate
from tqdm import tqdm
import pandas as pd


class Evaluation:
    def __init__(self, model,tokenizer,metric,model_name, evaluation_data, loop,language,device,backup_db):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.metric = metric
        self.model =model
        self.tokenizer = tokenizer
        self.database = backup_db
        self.evaluation_data = evaluation_data
        self.quesitons = None
        self.answers = None
        self.loop = loop
        self.scores = []
   
    
    def recall_qa(self,ask,q0):
        q_list= []
        a_list=[]
        q_list.append(q0)
        if self.language=='zh':
            prompt = '这个答案的问题最可能是什么:'
        elif self.language== 'en':
            prompt = 'What is the most likely question for this answer:'
        for i in range(self.loop):
            a = ask(self.model,self.tokenizer,self.device,q_list[i])
            a_list.append(a)
            q_next = ask(self.model,self.tokenizer,self.device,prompt+a)
            q_list.append(q_next)
        return q_list, a_list


    def recall_qas(self,ask):
            question0_list=self.evaluation_data
            question_records =[]
            answer_records =[]
            for i in tqdm(range(len(question0_list))):
                questions, answers = self.recall_qa(ask,question0_list[i])
                question_records.append(questions)
                answer_records.append(answers)
            return question_records, answer_records
            
            
    def evalutate(self,ask):
        self.quesitons,self.answers = self.recall_qas(ask)
        return self.quesitons,self.answers


    def get_score(self,mode = 'answer'):
        if mode == 'answer':
            records = self.answers
        elif mode == 'question':
            records = self.quesitons
        else:
            raise ValueError('mode should be answer or question')
        for i in range(self.loop):
            predictions = [record[0] for record in records]
            references = [record[i] for record in records]
            score = self.metric.compute(predictions=predictions,references=references)
            score.update({'loop':i,"refer":"0","mode":mode})
            self.scores.append(score)
            print(f"loop{i}:{score}")
        for i in range(1,self.loop):
            predictions = [answer[i-1] for answer in self.answers]
            references = [answer[i] for answer in self.answers]
            score = self.metric.compute(predictions=predictions,references=references)
            score.update({'loop':i,"refer":"n-1","mode":mode})
            self.scores.append(score)
            print(f"loop{i}:{score}")

        
    
    def write_scores_to_csv(self):
        df = pd.DataFrame(self.scores)
        df.to_csv(f'{self.model_name}_scores.csv', index=False)


    def write2db(self):
        for i in range(self.loop):
            rerord = {'model':self.model_name,'language':self.language,'question':self.quesitons[i],'answer':self.answers[i],'loop':self.loop}
            self.database.insert_one(rerord)
        