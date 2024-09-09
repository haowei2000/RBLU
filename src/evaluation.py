'''
a file that contains the Evaluation class, which is the main class for evaluating the model.
'''

import time
from typing import Callable, Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as torchDataset
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast

from process import Process


class TokenizedDataset(torchDataset):
    """
    TokenizedDataset is a custom dataset class for handling tokenized input data.

    Attributes:
        input_ids (torch.Tensor): A tensor containing the tokenized input IDs.
        attention_masks (torch.Tensor): A tensor containing the attention masks corresponding to the input IDs.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx: int): Returns a dictionary containing the input IDs and attention mask for the given index.
    """

    def __init__(self, input_ids: torch.Tensor, attention_masks: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
        }


class MyGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        batch_size,
        apply_template,
        gen_kwargs,
    ) -> None:
        self.model = model
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = tokenizer
        self.device = "cuda"
        self.max_new_token = 1024
        self.batch_size = batch_size
        self.apply_template: Optional[Callable[[str], list[dict]]] = apply_template
        self.gen_kwargs = gen_kwargs

    def __call__(self, text_list: list[str]) -> list[str]:
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
                    **model_inputs, gen_kwargs=self.gen_kwargs
                )
                response = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                responses.extend(response)
        end_time = time.time()
        print(f"Time taken for batch generation: {end_time - start_time:.2f} seconds")
        return responses

    def tokenize_texts(self, text_list: list[str]) -> BatchEncoding:
        if self.apply_template is not None:
            text_templated_list = [self.apply_template(text) for text in text_list]
            tokenized_batch = self.tokenizer.apply_chat_template(
                text_templated_list, tokenize=True, add_generation_prompt=True
            )
            if not isinstance(tokenized_batch, BatchEncoding):
                raise TypeError(
                    "The returned type from apply_chat_template must be BatchEncoding"
                )
        else:
            tokenized_batch = self.tokenizer.batch_encode_plus(
                text_list,
                padding="longest",
                return_tensors="pt",
            )
        return tokenized_batch


class Evaluation:
    def __init__(
        self,
        model,
        tokenizer,
        metric_compute: Callable[[list, list], dict],
        original_questions: list[str],
        batch_size: int,
        loop_count: int,
        apply_template: Optional[Callable[[str], list]] = None,
        process: Optional[Process] = None,
        gen_kwargs: Optional[dict] = None,
    ):
        self.generator = MyGenerator(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            apply_template=apply_template,
            gen_kwargs=gen_kwargs,
        )
        self.metric_compute = metric_compute
        self.qa_dataset = Dataset.from_dict({"q0": original_questions})
        self.loop_count = loop_count
        if process is None:
            self.process = Process()
        else:
            self.process = process

    def loop_evaluation(self):
        for loop in range(self.loop_count):
            print("Loop:", loop)
            self.qa_dataset = self.qa_dataset.map(
                self.process.question_template, fn_kwargs={"loop": loop}
            )
            self.qa_dataset = self.qa_dataset.add_column(
                name=f"a{loop}_output",
                column=self.generator(self.qa_dataset[f"q{loop}_prompt"]),
            )  # type: ignore
            self.qa_dataset = self.qa_dataset.map(
                self.process.answer_extract, fn_kwargs={"loop": loop}
            )
            self.qa_dataset = self.qa_dataset.map(
                self.process.answer_template, fn_kwargs={"loop": loop}
            )
            self.qa_dataset = self.qa_dataset.add_column(
                name=f"q{loop+1}_output",
                column=self.generator(self.qa_dataset[f"a{loop}_prompt"]),
            )  # type: ignore
            self.qa_dataset = self.qa_dataset.map(
                self.process.question_extract, fn_kwargs={"loop": loop}
            )

    def get_score(self, loop: int, mode: str, refer: str) -> dict:
        predictions, references = [], []
        if loop >= 1:
            if mode in ["q", "a"]:
                predictions = self.qa_dataset[f"{mode}{loop}"]
            else:
                print("mode error")
            if refer == "n-1":
                references = self.qa_dataset[f"{mode}{loop-1}"]
            elif refer == "0":
                references = self.qa_dataset[f"{mode}{0}"]
            else:
                print("Refer error")
        else:
            print("Loop error")
        score = self.metric_compute(predictions, references)
        return score
