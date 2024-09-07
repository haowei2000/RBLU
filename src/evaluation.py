"""
The `Record` class is used to save a record in the evaluation process,
the Chat` class is used for interacting with a language model,
and the `Evaluation` class is the main class for evaluating a Language Model.
"""

import dataclasses
import re
from typing import Callable
import datetime
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TokenizedDataset(Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_masks[idx]}


# it provides a default function to extract the question from a raw string
def default_question_extractor(question: str) -> str:
    """
    Extracts the default question from a given string.

    Parameters:
    q (str): The input string containing the question.

    Returns:
    str: The extracted default question.

    """
    match = re.search(r":\s*(.*)", question)
    if match:
        return match.group(1)
    return question


# it provides a default function to extract the question from a raw string
def default_answer_extractor(answer: str):
    """
    Extracts the default value of 'a'.

    Parameters:
        a (str): The input string.

    Returns:
        str: The extracted value of 'a'.
    """
    answer = answer.replace("Assistant:", "", 1)
    answer = answer.replace("assistant:", "", 1)
    return answer


def default_question_template(question: str) -> list:
    """
    Generates a default question template for a chatbot.

    Parameters:
    question (str): The question to be included in the template.

    Returns:
    list: A list of messages representing the template, with a system message and a user message.
    """
    messages = [
        # {"role": "system", "content": "You are a chatbot who can answer the question"},
        {"role": "user", "content": f"{question}"},
    ]
    return messages


def default_answer_template(answer: str) -> list:
    """
    Generates a default answer template for a chatbot.

    Parameters:
    answer (str): The answer for which the template is generated.

    Returns:
    list: A list of messages representing the default answer template. Each message is a dictionary with 'role' and 'content' keys.
    """
    messages = [
        # {
        #     "role": "system",
        #     "content": "You are a chatbot capable of guessing questions based on answers.",
        # },
        {"role": "user", "content": f"What is the most likely question for this answer:{answer}"},
    ]
    return messages


class Process:
    """
    a class to save the optional function in the evaluation,
    if not specified, we will create a default "Process"
    """

    def __init__(
        self,
        question_extract: Callable,
        answer_extract: Callable,
        question_template: Callable,
        answer_template: Callable,
    ) -> None:
        self.question_extract = question_extract
        self.answer_extract = answer_extract
        self.question_template = question_template
        self.answer_template = answer_template

    def batch_question_extract(self, text_list):
        return [self.question_extract(text) for text in text_list]

    def batch_answer_extract(self, text_list):
        return [self.answer_extract(text) for text in text_list]

    def batch_question_template(self, text_list):
        return [self.question_template(text) for text in text_list]

    def batch_answer_template(self, text_list):
        return [self.answer_template(text) for text in text_list]


class Chat:
    """
    a special chat class for llm evaluation
    """

    def __init__(self, model,tokenizer, gen_kwargs=None) -> None:
        """
        Initializes an instance of the Evaluation class.

        Args:
            model_checkpoint (str): The path or identifier of the pre-trained model checkpoint.

        Returns:
            None
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda"
        self.max_length_tokenized = 2048
        if gen_kwargs is None:
            self.gen_kwargs = {"max_new_tokens": 512}
        else:
            self.gen_kwargs = gen_kwargs

    def tokenize_texts(self, text_list):
        text_list = [self.tokenizer.apply_chat_template(text, tokenize=False) for text in text_list]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        tokenized_batch = self.tokenizer(
            text_list,
            padding='longest',
            # truncation=True,  # 截断到最大长度
            # max_length=self.max_length_tokenized,
            return_tensors="pt",  # 返回 PyTorch 张量
        )
        return tokenized_batch["input_ids"], tokenized_batch["attention_mask"]

    def generate_text(self, message: list):
        """
        Generates a response based on the given input.

        Parameters:
        - input (str): The input text to generate a response for.

        Returns:
        - str: The generated response.

        Raises:
        - None

        Example:
        ```
        response = ask("Hello, how are you?")
        ```
        """
        model_inputs = self.tokenizer.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        model_inputs["input_ids"] = model_inputs["input_ids"].to(self.device)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].to(self.device)
        # Directly use generate() and tokenizer.decode() to get the output.
        # Use `max_new_tokens` to control the maximum output length.
        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, **self.gen_kwargs)
            generated_ids = [
                output_ids[len(model_inputs.input_ids[i]) :]
                for i, output_ids in enumerate(generated_ids)
            ]

            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response

    def batch_generate_text(self, text_list, batch_size=2):
        """
        Generates a response based on the given input.

        Parameters:
        - input (str): The input text to generate a response for.

        Returns:
        - str: The generated response.

        Raises:
        - None

        Example:
        ```
        response = ask("Hello, how are you?")
        """
        input_ids, attention_masks = self.tokenize_texts(text_list)
        input_ids, attention_masks = input_ids.to(self.device), attention_masks.to(self.device)
        dataset = TokenizedDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        responses = []
        for model_inputs in dataloader:
            # Directly use generate() and tokenizer.decode() to get the output.
            # Use `max_new_tokens` to control the maximum output length.
            with torch.no_grad():
                generated_ids = self.model.generate(**model_inputs, **self.gen_kwargs)
                generated_ids = [
                    output_ids[len(model_inputs["input_ids"][i]) :]
                    for i, output_ids in enumerate(generated_ids)
                ]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                responses.extend(response)
        return responses


@dataclasses.dataclass
class Result:
    """
    it is 1 record to save the question, answer and result in the evaluation process
    """

    def __init__(
        self,
        loop,
        metric_compute,
    ):
        self.loop = loop
        self.scores_q_refer_0 = [None] * self.loop
        self.scores_q_refer_n = [None] * self.loop
        self.scores_a_refer_0 = [None] * self.loop
        self.scores_a_refer_n = [None] * self.loop
        self.questions = [None] * self.loop
        self.answers = [None] * self.loop
        self.metric_compute = metric_compute
        self.scores = None

    def get_score(self):
        for i in range(1, self.loop):
            self.scores_q_refer_0[i] = self.metric_compute(self.questions[0], self.questions[i])
            self.scores_q_refer_n[i] = self.metric_compute(self.questions[i - 1], self.questions[i])
            self.scores_a_refer_0[i] = self.metric_compute(self.answers[0], self.answers[i])
            self.scores_a_refer_n[i] = self.metric_compute(self.answers[i - 1], self.answers[i])
            df_q_refer_0 = pd.DataFrame(self.scores_q_refer_0)
            df_q_refer_0['type'] = 'q_refer_0'
            self.scores = df_q_refer_0
            df_a_refer_0 = pd.DataFrame(self.scores_a_refer_0)
            df_a_refer_0['type'] = 'a_refer_0'
            df_q_refer_n = pd.DataFrame(self.scores_q_refer_n)
            df_q_refer_n['type'] = 'q_refer_n'
            df_a_refer_n = pd.DataFrame(self.scores_a_refer_n)
            df_q_refer_n['type'] = 'a_refer_n'
            self.scores = pd.concat([self.scores, df_a_refer_0, df_q_refer_n, df_a_refer_n], axis=0)
    
    def save_score(self,path:str):
        self.scores.to_csv(path, index=False)

class Evaluation:
    """
    a main class to evaluation the LLM
    """

    def __init__(
        self,
        model,
        tokenizer,
        metric_compute,
        original_questions,
        loop,
        process,
    ):
        self.chat = Chat(model=model,tokenizer=tokenizer)
        self.metric_compute = metric_compute
        self.original_questions = original_questions
        self.loop = loop
        if process is None:
            self.process = Process(
                default_question_extractor,
                default_answer_extractor,
                default_question_template,
                default_answer_template,
            )
        else:
            self.process = process
        self.result = Result(loop=loop, metric_compute=metric_compute)

    def loop_evaluation(self):
        """
        Perform loop evaluation on a list of original questions.

        This method iterates over the original questions and performs a loop evaluation for each question.
        It uses the provided question and answer extractors, as well as the chat.ask method, to generate
        a list of questions and answers for each question in the loop.

        Returns:
            None

        Raises:
            None
        """
        self.result.questions[0] = self.original_questions
        for i in range(self.loop):
            print("Loop:", i)
            questions_list = self.process.batch_question_template(self.result.questions[i])
            # print(f'questions_list{i}:{questions_list}')
            answers_list_output = self.chat.batch_generate_text(questions_list)
            # print(f'answers_list_output{i}:{answers_list_output}')
            self.result.answers[i] = self.process.batch_answer_extract(answers_list_output)
            answers_list_input = self.process.batch_answer_template(self.result.answers[i])
            # print(f'answers_list_input{i}:{answers_list_input}')
            new_question_list_output = self.chat.batch_generate_text(answers_list_input)
            if i < self.loop - 1:
                self.result.questions[i + 1] = self.process.batch_question_extract(
                    new_question_list_output
                )

    def load_qa(self, questions: list[list], answers: list[list]):
        """
        Load a list of questions and answers into the object.
        Args:
            questions (list[list]): A list of question lists.
            answers (list[list]): A list of answer lists.
        Returns:
            None
        """
        self.result.questions = questions
        self.result.answers = answers

    def write_scores_to_csv(self, path: str):
        """
        Write the scores to a CSV file.
        Parameters:
        - task (str): The task name.
        Returns:
        - None
        """
        df = pd.DataFrame(self.result.scores)
        df.to_csv(path, index=False)
    
    def get_score(self):
        self.result.get_score()
    
    def save_score(self,path):
        self.result.save_score(path)

    # def write_qa2db(self, database):
    #     """
    #     Writes the question-answer pairs to a database.
    #     Args:
    #         database: The database object to write the records to.
    #     Returns:
    #         A list of records that were written to the database.
    #     """
    #     records = []
    #     if database is not None:
    #         for i in range(len(self.original_questions)):
    #             record = {
    #                 "question": self.result.questions[i],
    #                 "answer": self.result.answers[i],
    #                 "loop": self.loop,
    #                 "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #             }
    #             database.insert_one(record)
    #             records.append(record)
    #     return records
