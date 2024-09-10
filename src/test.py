# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import process

model_checkpoint = "/public/model/hub/llm-research/meta-llama-3-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
questions = ["Write me a poem about Machine Learning.", "Write me a poem about life."]
messages = [process.llama_template(question) for question in questions]
formatted_chats = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True,padding=True,tokenize=False
)
inputs = tokenizer(formatted_chats, return_tensors="pt",padding=True,add_special_tokens=False)  # type: ignore
inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
print("Tokenized inputs:\n", inputs)
outputs = model.generate(**inputs, max_new_tokens=256)
print("Generated tokens:\n", outputs)
decoded_outputs = [tokenizer.decode(output[inputs['input_ids'].size(1):], skip_special_tokens=True) for output in outputs]
print("Decoded outputs0:\n", decoded_outputs[0])
print("Decoded outputs1:\n", decoded_outputs[1])