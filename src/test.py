from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from proxy import set_proxy
# 加载模型和分词器
set_proxy()
model_name = "/public/model/hub/ZhipuAI/glm-4-9b-chat"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

# 输入文本列表
input_texts = ["translate English to French: Hello, how are you?",
"translate English to Spanish: I am fine, thank you."]
input_ids = tokenizer(input_texts, return_tensors="pt", padding=True).input_ids

# 批量生成
outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(output_texts)