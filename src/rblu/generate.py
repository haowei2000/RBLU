"""
The module `generate.py` contains a class `MyGenerator` that generates responses
based on the given model and tokenizer. The class has the following attributes:
"""

import json
import logging
from collections.abc import Callable
from time import time

import torch
from accelerate import Accelerator
from openai import OpenAI
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import (
    BatchEncoding,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from pymongo.collection import Collection as MongoCollection


class BackupGenerate:
    def __init__(
        self,
        backup_mongodb: MongoCollection,
        query_model_name: str,
        query_stage: str,
        query_language: str,
        query_task: str,
    ):
        self.mongodb = backup_mongodb
        self.query = {
            "model_name": query_model_name,
            "stage": query_stage,
            "language": query_language,
            "task": query_task,
        }

    def generate(self, text_list: list[str]) -> list[str]:
        return text_list

    def backup_generate(self, input: str) -> str:
        if search_result := self.mongodb.find_one(
            self.query | {"input": input}
        ):
            output = search_result["output"]
        else:
            output = self.select_generate(input)
            self.mongodb.insert_one(
                self.query | {"input": input, "output": output}
            )
        return output

    def __call__(self, text_list: list[str]) -> list[str]:
        response_list = [None] * len(text_list)
        unexisting_texts = []
        for i, text in enumerate(text_list):
            if self.backup_generate(text) is not None:
                response_list[i] = self.backup_generate(text)
            else:
                unexisting_texts.append((i, text))
        logging.info(
            "Number of responses to generate: %i (and %i existing in db)",
            len(unexisting_texts),
            len(text_list) - len(unexisting_texts),
        )
        responses_generated = self.generate(
            [text for _, text in unexisting_texts]
        )
        for i, response in enumerate(responses_generated):
            response_list[unexisting_texts[i][0]] = response
        if None in response_list:
            raise ValueError("Some responses were not generated.")
        return response_list


class TokenizedDataset(TorchDataset):
    """
    TokenizedDataset is a custom dataset class
    for handling tokenized input data.

    Attributes:
        input_ids (list or tensor): A list or tensor
        containing the tokenized input IDs.
        attention_masks (list or tensor): A list or tensor containing
        the attention masks corresponding to the input IDs.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns a dictionary containing the input IDs
        and attention mask for the given index.
    """

    def __init__(self, input_ids, attention_masks):
        super().__init__()
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
        }


class MyGenerator(BackupGenerate):
    """
    a class that with batch and chat template
    generates responses based on the given model and tokenizer
    """

    def __init__(
        self,
        model,
        tokenizer,
        batch_size,
        apply_template,
        tokenizer_kwargs,
        gen_kwargs,
        backup_mongodb,
        query_model_name,
        query_stage,
        query_language,
        query_task,
    ) -> None:
        """
        Initializes the generator class with the given parameters.

        Args:
            model:
                The model to be evaluated.
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast):
                The tokenizer associated with the model.
            batch_size (int):
                The number of samples to process in a batch.
            apply_template (Optional[Callable[[str], list[dict]]]):
                A function to apply a template to the input data.
            gen_kwargs (dict):
                Additional keyword arguments for the generation.

        Returns:
            None
        """
        super().__init__(
            backup_mongodb=backup_mongodb,
            query_model_name=query_model_name,
            query_stage=query_stage,
            query_language=query_language,
            query_task=query_task,
        )
        self.model = model
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            tokenizer
        )
        accelerator = Accelerator()
        model = accelerator.prepare(model)

        self.batch_size = batch_size
        self.apply_template: Callable[[str], list[dict]] | None = (
            apply_template
        )
        self.tokenizer_kwargs = tokenizer_kwargs
        self.gen_kwargs = gen_kwargs
        try:
            self.device = next(model.parameters()).device
        except StopIteration as e:
            raise ValueError("The model does not have any parameters.") from e

    def generate(self, text_list: list[str]) -> list[str]:
        """
        Generates responses for a list of input texts using the model.

        Args:
            text_list (list[str]):
                A list of input texts to be processed.

        Returns:
            list[str]: A list of generated responses corresponding to
            the input texts.

        This method performs the following steps:
        1. Tokenize the input texts.
        2. Converts the tokenized texts to the appropriate device.
        3. Creates a dataset and dataloader for batch processing.
        4. Generates responses using the model in a batched manner.
        5. Decodes the generated token IDs to strings.
        6. Measures and prints the time taken for the batch generation.
        """
        start_time = time()
        batch_encoding = self._tokenize_texts(text_list)
        dataset = TokenizedDataset(
            batch_encoding["input_ids"],
            batch_encoding["attention_mask"],
        )
        dataloader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False
        )

        responses = []
        for inputs in tqdm(dataloader, desc="Generating responses"):
            with torch.no_grad():
                inputs = {
                    key: tensor.to(self.device)
                    for key, tensor in inputs.items()
                }
                outputs = self.model.generate(**inputs, **self.gen_kwargs)
                decoded_outputs = self.tokenizer.batch_decode(
                    outputs[:, inputs["input_ids"].size(1) :],
                    skip_special_tokens=True,
                )
                responses.extend(decoded_outputs)
        logging.info(
            "Time taken for batch gen: %.2f seconds", time() - start_time
        )
        torch.cuda.empty_cache()
        return responses

    def _tokenize_texts(self, text_list: list[str]) -> BatchEncoding:
        """
        Tokenize a list of texts using the specified tokenizer.
        If a template is applied, it uses the `apply_chat_template`
        method of the tokenizer with additional options.
        Otherwise, it uses the `batch_encode_plus` method.

        Args:
            text_list (list[str]): A list of texts to be tokenized.

        Returns:
            BatchEncoding: The tokenized representation of the input
            texts.

        Raises:
            TypeError: If the returned type from `apply_chat_template`
            is not `BatchEncoding`.
        """
        if self.apply_template is not None:
            text_formatted_list = [
                self.apply_template(text) for text in text_list
            ]
            text_templated_list = self.tokenizer.apply_chat_template(
                text_formatted_list,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text_templated_list = [text_list]
        tokenized_batch = self.tokenizer.batch_encode_plus(
            text_templated_list,  # type: ignore
            return_tensors="pt",
            **self.tokenizer_kwargs,
        )
        if not isinstance(tokenized_batch, BatchEncoding):
            raise TypeError("The tokenized_batch is not `BatchEncoding`.")
        return tokenized_batch


class APIGenerator(BackupGenerate):
    def __init__(
        self,
        url: str,
        model_name: str,
        key: str,
        mongodb,
        query_model_name,
        query_stage,
        query_language,
        query_task,
    ):
        super().__init__(
            backup_mongodb=mongodb,
            query_model_name=query_model_name,
            query_stage=query_stage,
            query_language=query_language,
            query_task=query_task,
        )
        self.api_model_name = model_name
        self.url = url
        self.key = key

    @staticmethod
    def openai_generate(
        url: str, key: str, model_name: str, messages: list[dict]
    ):
        client = OpenAI(base_url=url, api_key=key)
        while True:
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )
                return completion.choices[0].message.content
            except (ConnectionError, TimeoutError) as e:
                logging.warning(
                    "Connection error occurred: %s. Retrying...", e
                )
                continue

    def select_generate(self, input: str) -> str:
        start_time = time()
        response = None
        match self.api_model_name:
            case "deepseekR1":
                response = self._deepseekR1_generate(input)
            case _ if "gpt" in self.api_model_name:
                response = self._chatgpt_generate(input)
            case _:
                raise ValueError("Invalid model name")
        logging.info(
            "Time taken for generation: %.2f seconds", time() - start_time
        )
        return response

    def _deepseekR1_generate(self, input: str):
        logging.info("Using DeepseekR1 model")
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {"role": "user", "content": input},
        ]
        url = "https://api.agicto.cn/v1"
        key = ""
        model = "deepseek-r1"
        return self.openai_generate(url, key, model, messages).split(
            "</think>\n\n"
        )[-1]

    def _chatgpt_generate(self, input: str) -> str:
        messages = [
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": input},
        ]
        return self.openai_generate(
            self.url, self.key, self.api_model_name, messages
        )

    def generate(self, text_list: list[str]) -> list[str]:
        responses = []
        responses.extend(
            self.select_generate(text)
            for text in tqdm(text_list, desc="Generating responses")
        )
        return responses
