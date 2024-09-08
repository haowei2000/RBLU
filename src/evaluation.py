"""
The `Record` class is used to save a record in the evaluation process,
the Chat` class is used for interacting with a language model,
and the `Evaluation` class is the main class for evaluating a Language Model.
"""

import time
from typing import Callable, Optional

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as torchDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from process import Process

class TokenizedDataset(torchDataset):
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
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
        }


class MyPipeline:
    """
    a special chat class for llm evaluation
    """

    def __init__(
        self,
        model,
        tokenizer,
        batch_size,
        template,
        gen_kwargs,
    ) -> None:
        """
        Initializes an instance of the Evaluation class.

        Args:
            model_checkpoint (str): The path or identifier of the pre-trained model checkpoint.

        Returns:
            None
        """
        self.model = model
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = tokenizer
        self.device = "cuda"
        self.max_length_tokenized = 2048
        self.max_new_token = 1024
        self.batch_size = batch_size
        self.template: Optional[Callable[[str], list[dict]]] = template

    def tokenize_texts(self, text_list):
        if self.template is not None:
            text_list = [self.template(text) for text in text_list]
            text_list = [self.tokenizer.apply_chat_template(text,tokenize=False) for text in text_list]
        tokenized_batch = self.tokenizer(
            text_list,
            padding="longest",
            # truncation=True,  # 截断到最大长度
            # max_length=self.max_length_tokenized,
            return_tensors="pt",
        )
        return tokenized_batch["input_ids"], tokenized_batch["attention_mask"]

    def __call__(self, text_list):
        start_time = time.time()
        input_ids, attention_masks = self.tokenize_texts(text_list)
        input_ids, attention_masks = (
            input_ids.to(self.device),
            attention_masks.to(self.device),
        )
        dataset = TokenizedDataset(input_ids, attention_masks)
        dataloader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False
        )
        responses = []

        for model_inputs in tqdm(dataloader, desc="Generating responses"):
            # Directly use generate() and tokenizer.decode() to get the output.
            # Use `max_new_tokens` to control the maximum output length.
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs, max_new_tokens=self.max_new_token
                )
                response = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                responses.extend(response)
        end_time = time.time()
        print(f"Time taken for batch generation: {end_time - start_time:.2f} seconds")

        return responses

class Evaluation:
    """
    a main class to evaluation the LLM
    """

    def __init__(
        self,
        model,
        tokenizer,
        metric_compute,
        original_questions: list[str],
        batch_size: int,
        loop_count: int,
        document_count: int,
        template: Optional[Callable] = None,
        process: Optional[Process] = None,
        gen_kwargs: Optional[dict] = None,
    ):
        self.generator = MyPipeline(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            template=template,
            gen_kwargs=gen_kwargs,
        )
        self.metric_compute = metric_compute
        self.qa_dataset = Dataset.from_dict({"q0": original_questions})
        self.loop_count = loop_count
        self.document_count = document_count
        if process is None:
            self.process = Process()
        else:
            self.process = process

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
        for loop in range(self.loop_count):
            print("Loop:", loop)
            self.qa_dataset = self.qa_dataset.map(
                self.process.question_template, fn_kwargs={"loop": loop}
            )
            self.qa_dataset=self.qa_dataset.add_column(
                name=f"a{loop}_output",
                column=self.generator(self.qa_dataset[f"q{loop}_prompt"]),
            )  # type: ignore
            self.qa_dataset = self.qa_dataset.map(
                self.process.answer_extract, fn_kwargs={"loop": loop}
            )
            self.qa_dataset = self.qa_dataset.map(
                self.process.answer_template, fn_kwargs={"loop": loop}
            )
            self.qa_dataset=self.qa_dataset.add_column(
                name=f"q{loop+1}_output", column=self.generator(self.qa_dataset[f"a{loop}_prompt"])
            )  # type: ignore
            self.qa_dataset=self.qa_dataset.map(self.process.question_extract, fn_kwargs={"loop": loop})
