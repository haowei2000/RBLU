"""
a file that contains the Evaluation class, which is the main class for evaluating the model.
"""

import time
from typing import Callable, Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast

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
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
        }


class MyGenerator:
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
    ) -> None:
        """
        Initializes the generator class with the given parameters.

        Args:
            model: The model to be evaluated.
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer associated with the model.
            batch_size (int): The number of samples to process in a batch.
            apply_template (Optional[Callable[[str], list[dict]]]): A function to apply a template to the input data.
            gen_kwargs (dict): Additional keyword arguments for the generation process.

        Returns:
            None
        """
        self.model = model
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = tokenizer
        self.batch_size = batch_size
        self.apply_template: Optional[Callable[[str], list[dict]]] = apply_template
        self.tokenizer_kwargs = tokenizer_kwargs
        self.gen_kwargs = gen_kwargs

    def __call__(self, text_list: list[str]) -> list[str]:
        """
        Generates responses for a list of input texts using the model.

        Args:
            text_list (list[str]): A list of input texts to be processed.

        Returns:
            list[str]: A list of generated responses corresponding to the input texts.

        This method performs the following steps:
        1. Tokenize the input texts.
        2. Converts the tokenized texts to the appropriate device.
        3. Creates a dataset and dataloader for batch processing.
        4. Generates responses using the model in a batched manner.
        5. Decodes the generated token IDs to strings.
        6. Measures and prints the time taken for the batch generation process.
        """
        start_time = time.time()
        batch_encoding = self.tokenize_texts(text_list)
        dataset = TokenizedDataset(batch_encoding["input_ids"], batch_encoding["attention_mask"])
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False)
        responses = []
        for inputs in tqdm(dataloader, desc="Generating responses"):
            # Directly use generate() and tokenizer.decode() to get the output.
            # Use `max_new_tokens` to control the maximum output length.
            with torch.no_grad():
                inputs = {key: tensor.to(self.model.device) for key, tensor in inputs.items()}
                outputs = self.model.generate(**inputs, **self.gen_kwargs)
                outputs = [
                    self.tokenizer.decode(
                        output[inputs["input_ids"].size(1) :], skip_special_tokens=True
                    )
                    for output in outputs
                ]
                responses.extend(outputs)
        end_time = time.time()
        print(f"Time taken for batch generation: {end_time - start_time:.2f} seconds")
        return responses

    def tokenize_texts(self, text_list: list[str]) -> BatchEncoding:
        """
        Tokenize a list of texts using the specified tokenizer. If a template is applied,
        it uses the `apply_chat_template` method of the tokenizer with additional options.
        Otherwise, it uses the `batch_encode_plus` method.

        Args:
            text_list (list[str]): A list of texts to be tokenized.

        Returns:
            BatchEncoding: The tokenized representation of the input texts.

        Raises:
            TypeError: If the returned type from `apply_chat_template` is not `BatchEncoding`.
        """
        if self.apply_template is not None:
            text_templated_list = [self.apply_template(text) for text in text_list]
            text_templated_list = self.tokenizer.apply_chat_template(
                text_templated_list,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text_templated_list = text_list
        tokenized_batch = self.tokenizer.batch_encode_plus(
            text_templated_list,  # type: ignore
            return_tensors="pt",
            **self.tokenizer_kwargs,
        )
        if not isinstance(tokenized_batch, BatchEncoding):
            raise TypeError("The tokenized_batch is not `BatchEncoding`.")
        return tokenized_batch


def evaluate(
    generator: Callable[[list[str]], list[str]],
    original_questions: list[str],
    loop_count: int,
    process: Process,
) -> Dataset:
    qa_dataset = Dataset.from_dict({"q0": original_questions})
    for loop in range(loop_count):
        print("Loop:", loop)
        qa_dataset = qa_dataset.map(process.question_template, fn_kwargs={"loop": loop})
        qa_dataset = qa_dataset.add_column(
            name=f"a{loop}_output",
            column=generator(qa_dataset[f"q{loop}_prompt"]),
        )  # type: ignore
        qa_dataset = qa_dataset.map(process.answer_extract, fn_kwargs={"loop": loop})
        qa_dataset = qa_dataset.map(process.answer_template, fn_kwargs={"loop": loop})
        qa_dataset = qa_dataset.add_column(
            name=f"q{loop+1}_output",
            column=generator(qa_dataset[f"a{loop}_prompt"]),
        )  # type: ignore
        qa_dataset = qa_dataset.map(process.question_extract, fn_kwargs={"loop": loop})
        return qa_dataset


def get_score(
    qa_dataset,
    metric_compute,
    loop: int,
    mode: str,
    refer: str,
) -> dict:
    """
    Computes the evaluation score based on the provided loop iteration, mode, and reference.

    Args:
        loop (int): The loop iteration. Must be greater than or equal to 1.
        mode (str): The mode of evaluation, either "q" for questions or "a" for answers.
        refer (str): The reference mode, either "n-1" to use the previous loop's data or "0" to use the initial data.

    Returns:
        dict: The computed score as a dictionary.

    Raises:
        ValueError: If the mode is not "q" or "a".
        ValueError: If the refer is not "n-1" or "0".
        ValueError: If the loop is less than 1.
    """
    predictions, references = [], []
    if loop >= 1:
        if mode in ["q", "a"]:
            predictions = qa_dataset[f"{mode}{loop}"]
        else:
            print("mode error")
        if refer == "n-1":
            references = qa_dataset[f"{mode}{loop-1}"]
        elif refer == "0":
            references = qa_dataset[f"{mode}{0}"]
        else:
            print("Refer error")
    else:
        print("Loop error")
    score = metric_compute(predictions, references)
    return score


def save_score(qa_dataset, metric_compute, loop_count, model_name, task, language, path):
    """
    Save the score to the disk.
    """
    scores = []
    for loop in range(loop_count):
        for mode in ["q", "a"]:
            for refer in ["n-1", "0"]:
                score = get_score(qa_dataset, metric_compute, loop, mode, refer)
                score["loop"] = loop
                score["refer"] = refer
                score["mode"] = mode
                score["model_name"] = model_name
                score["task"] = task
                score["language"] = language
                scores.append(score)
    scores = Dataset.from_dict(scores)
    scores.to_json(path, orient="records", lines=True)
    return qa_dataset
