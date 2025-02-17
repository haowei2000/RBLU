def process_batch_1(data):
    """处理第一批数据的函数"""
    return [f"{x}_A" for x in data]  # 示例处理：在元素后面加 `_A`

def process_batch_2(data):
    """处理第二批数据的函数"""
    return [f"{x}_B" for x in data]  # 示例处理：在元素后面加 `_B`

def split_and_process(lst):
    # 1️⃣ 拆分列表并保存原索引
    batch_1 = [x for i, x in enumerate(lst) if i % 2 == 0]  # 偶数索引的元素
    batch_2 = [x for i, x in enumerate(lst) if i % 2 != 0]  # 奇数索引的元素

    # 2️⃣ 送往不同的函数处理
    result_1 = process_batch_1(batch_1)
    result_2 = process_batch_2(batch_2)

    # 3️⃣ 结果合并，恢复原顺序
    result = []
    iter_1, iter_2 = iter(result_1), iter(result_2)
    for i in range(len(lst)):
        if i % 2 == 0:
            result.append(next(iter_1))  # 取 batch_1 结果
        else:
            result.append(next(iter_2))  # 取 batch_2 结果

    return result

# 测试数据
data_list = ["apple", "banana", "cherry", "date", "elderberry"]
final_result = split_and_process(data_list)
print(final_result)
