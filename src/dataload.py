import pandas as pd
import pymongo
import matplotlib.pyplot as plt

def load_field(field:str,count:int,min_length=50,max_length=100):
    if field == 'code':
        df = pd.read_parquet("hf://datasets/iamtarun/python_code_instructions_18k_alpaca/data/train-00000-of-00001-8b6e212f3e1ece96.parquet")
        df = df.rename(columns={"instruction": "question"})
        df['field'] = 'code'
    elif field in ['finance','law','medical']:
        collection = pymongo.MongoClient("10.48.48.7", 27017)["QA"]['backup_collection']
        records = collection.find({"area":field,'language':'en'},{'area':1,'question':1,'_id':0})
        records = [record for record in records if min_length<len(record['question']) < max_length]
        df = pd.DataFrame(records)
        df=df.rename(columns={'area':'field'})
    else:
        raise ValueError('Invalid field')
    df.drop_duplicates(subset='question', inplace=True)
    df = df.sample(n=count, random_state=42)
    df.to_csv(f'/workspace/project/llm_evaluation/data/{field}_{min_length}_{max_length}.csv', index=False)
    return df


def plot_string_length_distribution(data, key):
    data[key+'_length'] = data[key].str.len()
    histogram = data[key+'_length'].plot.hist(figsize=(10, 6), bins=20, range=(0, 200))
    histogram.set_xlabel('String Length')
    histogram.set_ylabel('Frequency')
    plt.show()


def main():
    from proxy import set_proxy
    set_proxy()
    # print(df)
    for field in ['code','medical','finance','law']:
        data = load_field(field,100,25,150)
        plot_string_length_distribution(data, 'question')



if __name__ == "__main__":
    main()