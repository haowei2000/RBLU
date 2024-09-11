# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import yaml
import process
with open("src/model_path.yml", "r", encoding="utf-8") as config_file:
    model_path = yaml.load(config_file, Loader=yaml.FullLoader)
model_checkpoint = model_path['chatglm']
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
questions = ["Write me a poem about life.","What is the problem when the dollar rises?", ]
# questions = [process.apply_gemma_template(question) for question in questions]
# questions = tokenizer.apply_chat_template(
#     questions, dd_generation_prompt=True,tokenize=False
# )
print(questions)
inputs = tokenizer(questions, return_tensors="pt",padding=True, truncation=True)  # type: ignore
inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
print("Tokenized inputs:\n", inputs)
outputs = model.generate(**inputs, max_new_tokens=256)
print("Generated tokens:\n", outputs)
decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
print("Decoded outputs0:\n", decoded_outputs[0])
print("Decoded outputs1:\n", decoded_outputs[1])