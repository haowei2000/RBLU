"""
The `Record` class is used to save a record in the evaluation process,
the Chat` class is used for interacting with a language model,
and the `Evaluation` class is the main class for evaluating a Language Model.
"""

import time
import datetime
import pandas as pd
import pymongo.collection
import torch
import pymongo
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from process import Process


class TokenizedDataset(Dataset):
    """
    TokenizedDataset is a custom dataset class for handling tokenized input data.

    Attributes:
        input_ids (list or tensor): A list or tensor containing the tokenized input IDs.
        attention_masks (list or tensor): A list or tensor containing the attention masks corresponding to the input IDs.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns a dictionary containing the input IDs and attention mask for the given index.
    """

    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_masks[idx]}


class Chat:
    """
    a special chat class for llm evaluation
    """

    def __init__(self, model, tokenizer, gen_kwargs=None) -> None:
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
            self.gen_kwargs = {"max_new_tokens": 2048}
        else:
            self.gen_kwargs = gen_kwargs

    def tokenize_texts(self, text_list):
        """
        Tokenizes a list of texts using the tokenizer associated with the instance.

        Args:
            text_list (list of str): List of texts to be tokenized.

        Returns:
            tuple: A tuple containing:
                - input_ids (torch.Tensor): Tensor of token ids.
                - attention_mask (torch.Tensor): Tensor of attention masks.
        """
        tokenized_batch = self.tokenizer(
            text_list,
            padding="longest",
            # truncation=True,  # 截断到最大长度
            # max_length=self.max_length_tokenized,
            return_tensors="pt",  # 返回 PyTorch 张量
        )
        return tokenized_batch["input_ids"], tokenized_batch["attention_mask"]

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

        start_time = time.time()

        input_ids, attention_masks = self.tokenize_texts(text_list)
        input_ids, attention_masks = input_ids.to(self.device), attention_masks.to(self.device)
        dataset = TokenizedDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        responses = []

        for model_inputs in tqdm(dataloader, desc="Generating responses"):
            # Directly use generate() and tokenizer.decode() to get the output.
            # Use `max_new_tokens` to control the maximum output length.
            with torch.no_grad():
                generated_ids = self.model.generate(**model_inputs, **self.gen_kwargs)
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                responses.extend(response)
        end_time = time.time()
        print(f"Time taken for batch generation: {end_time - start_time:.2f} seconds")

        return responses


# This class is named "Result" and likely contains methods or attributes related to storing or
# processing results.
class Result:
    """
    Initializes the evaluation class with the given parameters.

    Args:
        loop (int): The number of iterations or loops for the evaluation.
        metric_compute (callable): A function or callable object to compute the evaluation metric.

    Attributes:
        loop (int): The number of iterations or loops for the evaluation.
        scores_q_refer_0 (list): A list to store initial question reference scores for each loop.
        scores_q_refer_n (list): A list to store final question reference scores for each loop.
        scores_a_refer_0 (list): A list to store initial answer reference scores for each loop.
        scores_a_refer_n (list): A list to store final answer reference scores for each loop.
        questions (list): A list to store questions for each loop.
        answers (list): A list to store answers for each loop.
        metric_compute (callable): A function or callable object to compute the evaluation metric.
        scores (None or any): A placeholder for the computed scores, initially set to None.
    """

    def __init__(
        self,
        loop,
        document_count,
        metric_compute,
    ):
        self.document_count = document_count
        self.loop = loop
        self.scores_q_refer_0 = pd.DataFrame(index=range(loop - 1))
        self.scores_q_refer_n = pd.DataFrame(index=range(loop - 1))
        self.scores_a_refer_0 = pd.DataFrame(index=range(loop - 1))
        self.scores_a_refer_n = pd.DataFrame(index=range(loop - 1))
        self.questions = pd.DataFrame(index=range(loop), columns=range(document_count))
        self.answers = pd.DataFrame(index=range(loop), columns=range(document_count))
        self.metric_compute = metric_compute
        self.scores = None

    def get_score(self):
        """
        Computes and updates the scores for questions and answers based on a specified metric.
        This method iterates over a range of indices and computes scores for:
        - Questions compared to the first question.
        - Questions compared to the previous question.
        - Answers compared to the first answer.
        - Answers compared to the previous answer.
        The computed scores are stored in DataFrames with an additional 'type' column indicating the type of comparison.
        The final scores are concatenated into a single DataFrame.
        Attributes:
            self.scores_q_refer_0 (list): Scores for questions compared to the first question.
            self.scores_q_refer_n (list): Scores for questions compared to the previous question.
            self.scores_a_refer_0 (list): Scores for answers compared to the first answer.
            self.scores_a_refer_n (list): Scores for answers compared to the previous answer.
            self.scores (pd.DataFrame): DataFrame containing all the computed scores.
        Returns:
            None
        """
        q_refer_0, q_refer_n, a_refer_0, a_refer_n = [], [], [], []
        for i in range(1, self.loop):
            if (
                len(self.questions.loc[i]) != self.document_count
                or len(self.answers.loc[i]) != self.document_count
            ):
                print("length wrong")
            else:
                q_refer_0.append(self.metric_compute(self.questions.loc[0], self.questions.loc[i]))
                q_refer_n.append(
                    self.metric_compute(self.questions.loc[i - 1], self.questions.loc[i])
                )
                a_refer_0.append(self.metric_compute(self.answers.loc[0], self.answers.loc[i]))
                a_refer_n.append(self.metric_compute(self.answers.loc[i - 1], self.answers.loc[i]))
        self.scores_q_refer_0 = pd.DataFrame(q_refer_0)
        self.scores_a_refer_0 = pd.DataFrame(a_refer_0)
        self.scores_q_refer_n = pd.DataFrame(q_refer_n)
        self.scores_a_refer_n = pd.DataFrame(a_refer_n)
        self.scores_q_refer_0["loop"] = self.scores_q_refer_0.index
        self.scores_a_refer_0["loop"] = self.scores_a_refer_0.index
        self.scores_q_refer_n["loop"] = self.scores_q_refer_n.index
        self.scores_a_refer_n["loop"] = self.scores_a_refer_n.index
        self.scores_q_refer_0["type"] = "q_refer_0"
        self.scores_a_refer_0["type"] = "a_refer_0"
        self.scores_q_refer_n["type"] = "q_refer_n"
        self.scores_a_refer_n["type"] = "a_refer_n"
        self.scores = pd.concat(
            [
                self.scores_q_refer_0,
                self.scores_a_refer_0,
                self.scores_q_refer_n,
                self.scores_a_refer_n,
            ],
            axis=0,
        )

    def save_score(self, path: str):
        """
        Save the scores to a CSV file.

        Args:
            path (str): The file path where the CSV file will be saved.
        """
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
        batch_size,
        loop,
        document_count,
        process=None,
        apply_template=False,
        gen_kwargs=None,
    ):
        self.chat = Chat(model=model, tokenizer=tokenizer, gen_kwargs=gen_kwargs)
        self.metric_compute = metric_compute
        self.original_questions = original_questions
        self.batch_size = batch_size
        self.loop = loop
        self.document_count = document_count
        if process is None:
            self.process = Process(apply_template=apply_template,tokenizer=tokenizer)
        else:
            self.process = process
        self.result = Result(
            loop=loop, metric_compute=metric_compute, document_count=self.document_count
        )

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

        self.result.questions.loc[0] = self.original_questions
        for i in range(self.loop):
            print("Loop:", i)
            questions_old_raw_list = self.result.questions.loc[i, :].to_list()
            questions_old_template_list = self.process.batch_question_template(
                questions_old_raw_list
            )
            print(f'questions_old_template_list{i}:{questions_old_template_list[:2]}')
            answers_raw_list = self.chat.batch_generate_text(
                questions_old_template_list, batch_size=self.batch_size
            )
            print(f'answers_raw_list{i}:{answers_raw_list[:2]}')
            answers_extracted_list= self.process.batch_answer_extract(
                answers_raw_list, questions_old_template_list
            )
            self.result.answers.loc[i, :]=answers_extracted_list
            answers_template_list = self.process.batch_answer_template(
                self.result.answers.loc[i, :].to_list()
            )
            print(f'answers_template_list{i}:{answers_template_list[:2]}')
            questions_new_raw_list = self.chat.batch_generate_text(
                answers_template_list, batch_size=self.batch_size
            )
            print(f'questions_new_raw_list{i}:{questions_new_raw_list[:2]}')
            if i < self.loop - 1:
                new_question_extracted_list = self.process.batch_question_extract(
                    questions_new_raw_list, answers_template_list
                )
                self.result.questions.loc[i + 1, :] = new_question_extracted_list

    def load_from_db(self, database: pymongo.collection.Collection):
        """
        Load evaluation data from a MongoDB collection.

        This method retrieves records from the specified MongoDB collection where the "loop_count"
        matches the instance's loop attribute. It extracts the "question" and "answer" fields from
        each record and organizes them into pandas DataFrames, which are then assigned to the
        instance's result attribute.

        Args:
            database (pymongo.collection.Collection): The MongoDB collection from which to retrieve
                                                    the evaluation data.

        Returns:
            None
        """
        records = list(
            database.find({"loop_count": self.loop}, {"_id": 0, "question": 1, "answer": 1})
        )
        questions_data = {doc_order: record["question"] for doc_order, record in enumerate(records)}
        answers_data = {doc_order: record["answer"] for doc_order, record in enumerate(records)}
        questions_df = pd.DataFrame.from_dict(questions_data, orient="index").transpose()
        answers_df = pd.DataFrame.from_dict(answers_data, orient="index").transpose()
        self.result.questions = questions_df
        self.result.answers = answers_df
        return None

    def get_score(self):
        """
        Retrieves the score from the result attribute.

        This method calls the `get_score` method of the `result` attribute
        and returns its value.

        Returns:
            The score obtained from the `result` attribute.
        """
        self.result.get_score()

    def save_score(self, path):
        """
        Saves the evaluation score to the specified file path.

        Args:
            path (str): The file path where the score will be saved.
        """
        self.result.save_score(path)

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
            for doc_order in range(self.document_count):
                record = {
                    "question": self.result.questions.iloc[:, doc_order].to_list(),
                    "answer": self.result.answers.iloc[:, doc_order].to_list(),
                    "loop_count": self.loop,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                database.insert_one(record)
                records.append(record)
        return records
