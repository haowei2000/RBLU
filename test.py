
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def llama_ask(model, tokenizer, deivce, question):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=deivce,
        max_length=256,
    )
    answer = pipeline(question)
    return answer[0]['generated_text']

device = "auto"
model_id = "/public/model/Meta-Llama-3.1-8B/"
model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
question = "How can I improve my writing skills?"
model_inputs = tokenizer([question], return_tensors="pt")
generated_ids = model.generate(**model_inputs,max_new_tokens=512).cuda()
result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(result)
# print(llama_ask(model, tokenizer, 'auto', question))
# import transformers
# import torch

# model_id = "meta-llama/Meta-Llama-3.1-8B"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
#     max_length=256,
# )

# print(pipeline("Hey how are you doing today?"))
