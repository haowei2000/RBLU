'''
The `Record` class is used to save a record in the evaluation process,
the Chat` class is used for interacting with a language model,
and the `Evaluation` class is the mainclass for evaluating a Language Model.
'''
import dataclasses
import re
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
def default_answer_extractor(answer: str) -> str:
    """
    Extracts the default value of 'a'.

    Parameters:
        a (str): The input string.

    Returns:
        str: The extracted value of 'a'.
    """
    return answer


# it is 1 record to save the question, answer and result in the evaluation process
@dataclasses.dataclass
class Record:
    """
    1 record in LLM evaluation
    """

    def __init__(
        self,
        original_question,
        loop,
        qa_method,
        question_extractor,
        answer_extractor,
        q_prompt,
        a_prompt,
    ):
        self.original_question = original_question
        self.questions_loop = [original_question]
        self.answers_loop = []
        self.qa_method = qa_method
        self.loop = loop
        self.question_extractor = question_extractor
        self.answer_extractor = answer_extractor
        self.q_prompt = q_prompt
        self.a_prompt = a_prompt

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
        # q_prompt = "What is the most likely question for this answer:"
        for i in range(self.loop):
            old_question = self.questions_loop[i]
            answer = self.qa_method(f"{self.q_prompt}{old_question}")
            answer = self.answer_extractor(answer)
            self.answers_loop.append(answer)
            new_question = self.qa_method(f"{self.a_prompt}{answer}")
            new_question = self.question_extractor(new_question)
            self.questions_loop.append(new_question)


@dataclasses.dataclass
class Chat:
    """
    a special chat class for llm evaluation
    """

    def __init__(self, model_checkpoint) -> None:
        """
        Initializes an instance of the Evaluation class.

        Args:
            model_checkpoint (str): The path or identifier of the pre-trained model checkpoint.

        Returns:
            None
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.device = "cuda"

    def ask(self, input_text: str):
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
        print(response)
        ```
        """
        input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0])


class Evaluation:
    """
    a main class to evaluation the LLM
    """

    def __init__(
        self,
        model_checkpoint,
        metric,
        original_questions,
        loop,
        q_extractor=None,
        a_extractor=None,
    ):
        self.model_checkpoint = model_checkpoint
        self.chat = Chat(model_checkpoint)
        self.metric = metric
        self.original_questions = original_questions
        self.loop = loop
        self.questions_list = []
        self.answers_list = []
        self.scores = []
        self.question_prompt = "What is the most likely question for this answer:"
        self.answer_prompt = ""
        if q_extractor is None:
            self.q_extractor = default_question_extractor
        else:
            self.q_extractor = q_extractor
        if a_extractor is None:
            self.a_extractor = default_answer_extractor
        else:
            self.a_extractor = a_extractor

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
                question_extractor=self.q_extractor,
                answer_extractor=self.a_extractor,
                q_prompt=self.question_prompt,
                a_prompt=self.answer_prompt,
            )
            record.qa_loop()
            self.questions_list.append(record.questions_loop)
            self.answers_list.append(record.answers_loop)

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
            records = self.answers_list
        elif mode == "question":
            records = self.questions_list
        else:
            raise ValueError("mode should be answer or question")
        for i in range(self.loop):
            predictions = [record[0] for record in records]
            references = [record[i] for record in records]
            score = self.metric.compute(predictions=predictions, references=references)
            score.update({"loop": i, "refer": "0", "mode": mode})
            self.scores.append(score)
            print(f"loop{i}:{score}")
        for i in range(1, self.loop):
            predictions = [record[i - 1] for record in records]
            references = [record[i] for record in records]
            score = self.metric.compute(predictions=predictions, references=references)
            score.update({"loop": i, "refer": "n-1", "mode": mode})
            self.scores.append(score)
            print(f"loop{i}:{score}")

    def load_qa(self, questions: list[list], answers: list[list]):
        """
        Load a list of questions and answers into the object.

        Args:
            questions (list[list]): A list of question lists.
            answers (list[list]): A list of answer lists.

        Returns:
            None
        """
        self.questions_list = questions
        self.answers_list = answers

    def write_scores_to_csv(self, task: str):
        """
        Write the scores to a CSV file.

        Parameters:
        - task (str): The task name.

        Returns:
        - None
        """
        df = pd.DataFrame(self.scores)
        df.to_csv(f"score/{self.model_checkpoint}_{task}_scores.csv", index=False)

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
                    "question": self.questions_list[i],
                    "answer": self.answers_list[i],
                    "loop": self.loop,
                }
                database.insert_one(record)
                records.append(record)
        return records
