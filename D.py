from collections import Counter

def find_max_lists(tuples):
    def get_sorted_keys(lst):
        # 获取数字及其频率
        count = Counter(lst)
        # 根据频率和数字大小排序，确保先处理较大的对
        sorted_items = sorted(count.items(), key=lambda x: (-x[1], -x[0]))
        # 获取成对的数字和单独的数字
        pairs = [item[0] for item in sorted_items if item[1] == 2]
        single = [item[0] for item in sorted_items if item[1] == 1][0]
        # 返回排序后的元素列表：先较大的对，然后较小的对，最后单独的数字
        return pairs + [single] if len(pairs) == 2 else None

    max_lists = []
    max_keys = None

    # 遍历所有元组，比较和筛选
    for identifier, lst in tuples:
        sorted_keys = get_sorted_keys(lst)
        if sorted_keys:
            if max_lists == [] or sorted_keys > max_keys:
                max_lists = [(identifier, lst)]
                max_keys = sorted_keys
            elif sorted_keys == max_keys:
                max_lists.append((identifier, lst))

    return max_lists

# 示例
tuples = [
    (1, [2, 2, 3, 3, 1]),
    (2, [4, 4, 3, 3, 2]),
    (3, [2, 2, 5, 5, 1]),
    (4, [3, 3, 4, 4, 5]),
    (5, [4, 4, 6, 6, 1]),
    (5, [4, 4, 6, 6, 1])
]

result = find_max_lists(tuples)
print("最大的列表或列表组是：")
for identifier, lst in result:
    print(identifier, lst)
