"""
a script to load data from different sources and save it to csv
folder path is ./data
"""

import pandas as pd
import pymongo
import matplotlib.pyplot as plt
from proxy import set_proxy


def load_field(
    field: str, count: int, min_length: int = 50, max_length: int = 100, from_remote=True
) -> list:
    """
    Load data based on the specified field.

    Args:
        field (str): The field to load data for. Can be "code", "finance", "law", or "medical".
        count (int): The number of records to load.
        min_length (int, optional): The minimum length of the question. Defaults to 50.
        max_length (int, optional): The maximum length of the question. Defaults to 100.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.

    Raises:
        ValueError: If an invalid field is specified.
    """
    result = []
    path= f"./data/{field}_{count}_{min_length}_{max_length}.csv"
    if from_remote:
        if field == "code":
            df = pd.read_parquet(
                "hf://datasets/iamtarun/python_code_instructions_18k_alpaca/data/train-00000-of-00001-8b6e212f3e1ece96.parquet"
            )
            df = df.rename(columns={"instruction": "question"})
            df["field"] = "code"
        elif field in ["finance", "law", "medical"]:
            collection = pymongo.MongoClient("10.48.48.7", 27017)["QA"]["backup_collection"]
            records = collection.find(
                {"area": field, "language": "en"}, {"area": 1, "question": 1, "_id": 0}
            )
            records = [
                record for record in records if min_length < len(record["question"]) < max_length
            ]
            df = pd.DataFrame(records)
            df = df.rename(columns={"area": "field"})
        else:
            raise ValueError("Invalid field")
        df.drop_duplicates(subset="question", inplace=True)
        if len(df) < count:
            raise ValueError(f"Not enough data to sample {count} records for field {field}")
        df = df.sample(n=count, random_state=42)
        df.to_csv(path, index=False)
        result = df["question"].to_list()
    else:
        result = pd.read_csv(path)["question"].to_list()
    return result


def plot_string_length_distribution(data: pd.DataFrame, key: str) -> None:
    """
    Plots the distribution of string lengths in a given DataFrame column.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        key (str): The column name in the DataFrame.

    Returns:
        None
    """
    data[key + "_length"] = data[key].str.len()
    histogram = data[key + "_length"].plot.hist(figsize=(10, 6), bins=20, range=(0, 200))
    histogram.set_xlabel("String Length")
    histogram.set_ylabel("Frequency")
    plt.show()


def main():
    """
    This is the main function that loads data for different fields and
    plots the string length distribution.

    Parameters:
    None

    Returns:
    None
    """
    set_proxy()
    # print(df)
    for field in ["code", "medical", "finance", "law"]:
        data = load_field(field, 100, 25, 150)
        plot_string_length_distribution(data, "question")


if __name__ == "__main__":
    main()
