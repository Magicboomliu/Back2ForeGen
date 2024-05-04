from collections import Counter


def compare_EQ8(list1, list2):
    c1 = max(list1)
    c2 = max(list2)
    if c1>c2:
        return True
    else:
        return False
    

def compare_EQ7(list1, list2):
    def find_common_and_unique(cards):
        count = Counter(cards)
        common_card = unique_card = None
        # 寻找四张相同的和一张不同的牌
        for card, cnt in count.items():
            if cnt == 4:
                common_card = card
            else:
                unique_card = card
        return int(common_card), int(unique_card)  # 将牌面转为整数以便比较

    # 获取两个列表中的常见牌和独特牌
    common1, unique1 = find_common_and_unique(list1)
    common2, unique2 = find_common_and_unique(list2)

    # 先比较四张相同的牌
    if common1 > common2:
        return True
    elif common1 < common2:
        return False
    else:
        # 如果四张相同的牌相等，比较不同的那张牌
        return unique1 > unique2
    

def compare_EQ6(hand1, hand2):
    def analyze_hand(hand):
        """ 分析手牌，返回三条和对子的值 """
        count = Counter(hand)
        three_of_a_kind = None
        pair = None
        # 遍历计数器，找出三条和对子
        for card, cnt in count.items():
            if cnt == 3:
                three_of_a_kind = int(card)  # 假设牌面已经是整数或可直接转换为整数
            elif cnt == 2:
                pair = int(card)
        return three_of_a_kind, pair

    # 分析两手牌
    three1, pair1 = analyze_hand(hand1)
    three2, pair2 = analyze_hand(hand2)

    # 按规则比较三条的大小
    if three1 > three2:
        return True
    elif three1 < three2:
        return False
    else:
        # 如果三条相等，比较对子的大小
        return pair1 > pair2


def compared_EQ5(hand1,hand2):
    return hand1>hand2



def compared_EQ4(hand1,hand2):
    pass

def compared_EQ3(hand1,hand2):
    pass

def compared_EQ2(hand1,hand2):
    pass

def compared_EQ1(hand1,hand2):
    return hand1>hand2

