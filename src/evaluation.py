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
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

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


@dataclasses.dataclass
class Process:
    """
    a class to save the optional function in the evaluation,
    if not specified, we will create a default "Process"
    """

    question_extract: Callable
    answer_extract: Callable
    question_template: Callable
    answer_template: Callable


@dataclasses.dataclass
class Chat:
    """
    a special chat class for llm evaluation
    """

    def __init__(self, model_checkpoint,gen_kwargs=None) -> None:
        """
        Initializes an instance of the Evaluation class.

        Args:
            model_checkpoint (str): The path or identifier of the pre-trained model checkpoint.

        Returns:
            None
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
        self.device = "cuda"
        if gen_kwargs is None:
            self.gen_kwargs = {"max_new_tokens": 256}
        else:
            self.gen_kwargs = gen_kwargs

    def ask(self, messages: list):
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
        # self.model = deepspeed.init_inference(
        #     self.model,
        #     tensor_parallel={"tp_size": 2},
        #     dtype=torch.half,
        #     checkpoint=None,
        #     replace_with_kernel_inject=True
        # )
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
        model_inputs["input_ids"] = model_inputs["input_ids"].to(self.device)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].to(self.device)

        # Directly use generate() and tokenizer.decode() to get the output.
        # Use `max_new_tokens` to control the maximum output length.
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                **self.gen_kwargs
            )
            generated_ids = [
                output_ids[len(model_inputs.input_ids[i]) :]
                for i, output_ids in enumerate(generated_ids)
            ]

            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response
        # input_ids = self.tokenizer.apply_chat_template(input_text, return_tensors="pt").to(self.device)
        # outputs = self.model.generate(
        #     input_ids,
        #     max_new_tokens=512,
        # )
        # return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


@dataclasses.dataclass
class Result:
    """
    a class to save the result of the evaluation
    """

    questions: list
    answers: list
    scores: list


@dataclasses.dataclass
class Record:
    """
    it is 1 record to save the question, answer and result in the evaluation process
    """

    def __init__(
        self,
        original_question,
        loop,
        qa_method,
        process: Process,
    ):
        self.original_question = original_question
        self.qa_method = qa_method
        self.loop = loop
        self.process = process
        self.questions_loop = [original_question]
        self.answers_loop = []

    def qa_loop(self):
        """
        Generates a list of questions and answers based on a given initial question.

        Args:
            ask (function): A function that takes a model, tokenizer, device, and a question
            as input and returns an answer.
            q0 (str): The initial question.

        Returns:
            tuple: A tuple containing two lists - q_list and a_list.
                - q_list (list): A list of questions generated during the process.
                - a_list (list): A list of answers corresponding to the generated questions.
        """
        for i in range(self.loop):
            old_question = self.questions_loop[i]
            answer = self.qa_method(self.process.question_template(old_question))
            answer = self.process.answer_extract(answer)
            self.answers_loop.append(answer)
            new_question = self.qa_method(self.process.answer_template(answer))
            new_question = self.process.question_extract(new_question)
            self.questions_loop.append(new_question)


class Evaluation:
    """
    a main class to evaluation the LLM
    """

    def __init__(
        self,
        model_checkpoint,
        metric_compute,
        original_questions,
        loop,
        process,
        chat=None,
    ):
        self.model_checkpoint = model_checkpoint
        if chat is None:
            self.chat = Chat(model_checkpoint)
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
        self.result = Result([], [], [])

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
        for i in tqdm(range(len(self.original_questions))):
            original_question = self.original_questions[i]
            record = Record(
                original_question=original_question,
                loop=self.loop,
                qa_method=self.chat.ask,
                process=self.process,
            )
            record.qa_loop()
            self.result.questions.append(record.questions_loop)
            self.result.answers.append(record.answers_loop)

    def get_score(self, mode: str = "answer"):
        """
        Calculate the score based on the given mode.

        Parameters:
        - mode (str): The mode to calculate the score. Default is "answer".

        Raises:
        - ValueError: If the mode is neither "answer" nor "question".

        Returns:
        - None
        """
        if mode == "answer":
            records = self.result.answers
        elif mode == "question":
            records = self.result.questions
        else:
            raise ValueError("mode should be answer or question")
        for i in range(1,self.loop):
            predictions = [record[0] for record in records]
            references = [record[i] for record in records]
            score = self.metric_compute(predictions=predictions, references=references)
            score.update({"loop": i, "refer": "0", "mode": mode})
            self.result.scores.append(score)
        for i in range(1, self.loop):
            predictions = [record[i - 1] for record in records]
            references = [record[i] for record in records]
            score = self.metric_compute(predictions=predictions, references=references)
            score.update({"loop": i, "refer": "n-1", "mode": mode})
            self.result.scores.append(score)
        return self.result.scores

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

    def write_qa2db(self, database):
        """
        Writes the question-answer pairs to a database.

        Args:
            database: The database object to write the records to.

        Returns:
            A list of records that were written to the database.
        """
        records = []
        if database is not None:
            for i in range(len(self.original_questions)):
                record = {
                    "model": self.model_checkpoint,
                    "question": self.result.questions[i],
                    "answer": self.result.answers[i],
                    "loop": self.loop,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                database.insert_one(record)
                records.append(record)
        return records
