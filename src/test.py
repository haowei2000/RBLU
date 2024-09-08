from datasets import Dataset
my_dict = {"a": [1, 2, 3],"b":[3,2,1]}
dataset = Dataset.from_dict(my_dict)
def mult(example,name):
    example[name] = example[name]*example[name]
    return example
dataset=dataset.map(mult,fn_kwargs={"name":"a"})
print(dataset['a'])

