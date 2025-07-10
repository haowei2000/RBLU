"""
A module to generate reverse dataset
"""

from datasets import Dataset, load_from_disk

from rblu.utils.path import RESULT_DIR

model_name, area, argument_type, language = "glm", "financial", "reverse", "en"
data_path = RESULT_DIR / f"{model_name}_{area}_{argument_type}_{language}"
max_loop_count = 5
argued_dataset: Dataset = load_from_disk(data_path)
print(argued_dataset)
dataset2train: list = []
for loop in range(5):
    question_column, answer_column = f"a{loop}_prompt", f"q{loop + 1}_output"
    train_dataset = argued_dataset.select_columns(
        [question_column, answer_column]
    )
    sample_record = train_dataset.to_pandas().head(1)
    print(sample_record)
