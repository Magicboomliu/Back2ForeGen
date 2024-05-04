from collections import Counter

def card_value(card):
    """ Return the numerical value of a card (2-14, where Ace is 14). """
    values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
              '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    return values[card[1]]

def rank_hand(hand):
    """ Rank the given poker hand. """
    if not hand:
        return 0

    # Sort cards by value, considering Ace as both high and low for straights
    sorted_hand = sorted(hand, key=card_value)
    high_values = [card_value(card) for card in sorted_hand]
    low_values = [1 if value == 14 else value for value in high_values]  # Ace as 1 for low straights

    # Check for flush
    is_flush = len(set(card[0] for card in hand)) == 1

    # Function to check for straight
    def is_straight(values):
        return len(set(values)) == 5 and (max(values) - min(values) == 4)

    # Check for high and low straights
    is_high_straight = is_straight(high_values)
    is_low_straight = is_straight(low_values)

    # Assign the correct values list based on straight type
    values = high_values if is_high_straight else low_values if is_low_straight else sorted(high_values, reverse=True)
    value_counts = Counter(values).most_common()

    # Ranking hands
    if (is_high_straight or is_low_straight) and is_flush:
        return (8, values)  # Straight flush
    elif value_counts[0][1] == 4:
        return (7, values)  # Four of a kind
    elif value_counts[0][1] == 3 and value_counts[1][1] == 2:
        return (6, values)  # Full house
    elif is_flush:
        return (5, values)  # Flush
    elif is_high_straight or is_low_straight:
        return (4, values)  # Straight
    elif value_counts[0][1] == 3:
        return (3, values)  # Three of a kind
    elif value_counts[0][1] == 2 and value_counts[1][1] == 2:
        return (2, values)  # Two pairs
    elif value_counts[0][1] == 2:
        return (1, values)  # One pair
    else:
        return (0, values)  # High card

def is_straight(values):
    return len(set(values)) == 5 and (max(values) - min(values) == 4)

def get_rank_of_hand(hand):
    
    # sorted the cards
    sequences = hand.split()    
    sorted_hand = sorted(sequences, key=card_value)
    
    
    
    # get the current ranks
    high_values = [card_value(card) for card in sorted_hand]
    low_values = [1 if value == 14 else value for value in high_values]
    
        
    # Check for flush
    is_flush = len(set(card[0] for card in sorted_hand)) == 1

    

    # Check for high and low straights
    is_high_straight = is_straight(high_values)
    is_low_straight = is_straight(low_values)
    
    values = high_values if is_high_straight else low_values if is_low_straight else sorted(high_values, reverse=True)
    
    value_counts = Counter(values).most_common()

    # Ranking hands
    if (is_high_straight or is_low_straight) and is_flush:
        return (8, values)  # Straight flush
    elif value_counts[0][1] == 4:
        return (7, values)  # Four of a kind
    elif value_counts[0][1] == 3 and value_counts[1][1] == 2:
        return (6, values)  # Full house
    elif is_flush:
        return (5, values)  # Flush
    elif is_high_straight or is_low_straight:
        return (4, values)  # Straight
    elif value_counts[0][1] == 3:
        return (3, values)  # Three of a kind
    elif value_counts[0][1] == 2 and value_counts[1][1] == 2:
        return (2, values)  # Two pairs
    elif value_counts[0][1] == 2:
        return (1, values)  # One pair
    else:
        return (0, values)  # High card
    


def select_the_highest_rank_hands(types_and_values):
    selectived_ranks =[]
    highest_orderers = max([rank[0] for rank in types_and_values])
    
    for value in types_and_values:
        if value[0] == highest_orderers:
            selectived_ranks.append(value)
    
    return selectived_ranks

def find_numbers4(nums):
    # 计算每个数字出现的次数
    count = Counter(nums)
    
    # 初始化变量
    num_appears_four = None
    num_appears_once = None
    
    # 遍历统计结果，分别找到出现四次和一次的数字
    for num, freq in count.items():
        if freq == 4:
            num_appears_four = num
        elif freq == 1:
            num_appears_once = num
    
    return num_appears_four, num_appears_once

def find_numbers3(nums):
    # 计算每个数字出现的次数
    count = Counter(nums)
    
    # 初始化变量
    num_appears_four = None
    num_appears_once = None
    
    # 遍历统计结果，分别找到出现四次和一次的数字
    for num, freq in count.items():
        if freq == 3:
            num_appears_four = num
        elif freq == 2:
            num_appears_once = num
    
    return num_appears_four, num_appears_once


def get_the_higest_hand(selective_hands):
    '''
    input: selective hands
    output: higest hands and how many it have
    
    '''
    highest_hands = []
    if selective_hands[0][0]==8:
        higgest_score = max([hand[1][0] for hand in selective_hands])
        for hand in selective_hands:
            if hand[1][0] == higgest_score:
                highest_hands.append(hand)
        
    if selective_hands[0][0]==7:
        the_four_list =[]
        selective_four_list = []
        for hand in selective_hands:
            the_four, the_one = find_numbers4(hand[1])
            the_four_list.append([the_four,the_one])
        biggest_four = max([four[0] for four in the_four_list])
        for hand in selective_hands:
            the_four, the_one = find_numbers4(hand[1])
            if the_four ==biggest_four:
                selective_four_list.append(hand)
        if len(selective_four_list)!=1:
            the_four_one_list = []
            for hand in selective_four_list:
                the_one = find_numbers4(hand[1])[1]
                the_four_one_list.append(the_one)
            biggest_one = max(the_four_one_list)
            for hand in selective_four_list:
                the_one = find_numbers4(hand[1])[1]
                if the_one==biggest_one:
                    highest_hands.append(hand)
        else:
            highest_hands = selective_four_list
            
            
    if selective_hands[0][0]==6:
        the_four_list =[]
        selective_four_list = []
        for hand in selective_hands:
            the_four, the_one = find_numbers3(hand[1])
            the_four_list.append([the_four,the_one])
        biggest_four = max([four[0] for four in the_four_list])
        for hand in selective_hands:
            the_four, the_one = find_numbers3(hand[1])
            if the_four ==biggest_four:
                selective_four_list.append(hand)
        if len(selective_four_list)!=1:
            the_four_one_list = []
            for hand in selective_four_list:
                the_one = find_numbers3(hand[1])[1]
                the_four_one_list.append(the_one)
            biggest_one = max(the_four_one_list)
            for hand in selective_four_list:
                the_one = find_numbers3(hand[1])[1]
                if the_one==biggest_one:
                    highest_hands.append(hand)
        else:
            highest_hands = selective_four_list
            
            
    if selective_hands[0][0]==5:
        max_lists = []
        # 遍历所有列表
        for lst in selective_hands:
            # 如果max_lists为空，添加当前列表
            if not max_lists:
                max_lists.append(lst)
            else:
                # 比较当前列表与max_lists中已有列表的字典序
                if lst[1] > max_lists[0][1]:
                    # 如果当前列表更大，清空max_lists并添加当前列表
                    max_lists = [lst]
                elif lst[1] == max_lists[0][1]:
                    # 如果当前列表与最大列表相等，添加当前列表
                    max_lists.append(lst)
        
        highest_hands = max_lists
        
        
        
    if selective_hands[0][0]==4:
        higgest_score = max([hand[1][0] for hand in selective_hands])
        for hand in selective_hands:
            if hand[1][0] == higgest_score:
                highest_hands.append(hand)
                
    if selective_hands[0][0]==3:
        def identify_numbers(lst):
            count = Counter(lst)
            three_times = None
            two_diffs = []
            for num, freq in count.items():
                if freq == 3:
                    three_times = num
                elif freq == 1:
                    two_diffs.append(num)
            two_diffs.sort(reverse=True)
            return three_times, two_diffs

        max_lists = []
        max_three_times = None
        max_two_diffs = None

        # 遍历所有元组，元组的第二个元素是我们需要比较的列表
        for identifier, lst in selective_hands:
            three_times, two_diffs = identify_numbers(lst)
            if max_lists == [] or three_times > max_three_times or \
            (three_times == max_three_times and two_diffs > max_two_diffs):
                max_lists = [(identifier, lst)]
                max_three_times = three_times
                max_two_diffs = two_diffs
            elif three_times == max_three_times and two_diffs == max_two_diffs:
                max_lists.append((identifier, lst))
        
        highest_hands = max_lists
                
    if selective_hands[0][0]==2:
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
        for identifier, lst in selective_hands:
            sorted_keys = get_sorted_keys(lst)
            if sorted_keys:
                if max_lists == [] or sorted_keys > max_keys:
                    max_lists = [(identifier, lst)]
                    max_keys = sorted_keys
                elif sorted_keys == max_keys:
                    max_lists.append((identifier, lst))
        
        highest_hands = max_lists 
        
    if selective_hands[0][0]==1:
        def identify_key_numbers(lst):
            count = Counter(lst)
            duplicate = None
            singles = []
            for num, freq in count.items():
                if freq == 2:
                    duplicate = num
                elif freq == 1:
                    singles.append(num)
            singles.sort(reverse=True)  # 从大到小排序这三个不一样的数字
            return duplicate, singles

        max_lists = []
        max_duplicate = None
        max_singles = None

        # 遍历所有元组，元组的第二个元素是我们需要比较的列表
        for identifier, lst in selective_hands:
            duplicate, singles = identify_key_numbers(lst)
            if max_lists == [] or duplicate > max_duplicate or \
            (duplicate == max_duplicate and singles > max_singles):
                max_lists = [(identifier, lst)]
                max_duplicate = duplicate
                max_singles = singles
            elif duplicate == max_duplicate and singles == max_singles:
                max_lists.append((identifier, lst))
        highest_hands = max_lists
        
        
    
    if selective_hands[0][0]==0:   
        max_lists = []
        # 遍历所有列表
        for lst in selective_hands:
            # 如果max_lists为空，添加当前列表
            if not max_lists:
                max_lists.append(lst)
            else:
                # 比较当前列表与max_lists中已有列表的字典序
                if lst[1] > max_lists[0][1]:
                    # 如果当前列表更大，清空max_lists并添加当前列表
                    max_lists = [lst]
                elif lst[1] == max_lists[0][1]:
                    # 如果当前列表与最大列表相等，添加当前列表
                    max_lists.append(lst)
        highest_hands = max_lists
    
    return highest_hands







def compared_two_hands(hands1,hands2):
    
    "return rank1 is bigger than rank2 or not"
    rank1 = hands1[0]
    rank2 = hands2[0]
    value1 = hands1[1]
    value2 = hands2[1]
    
    if rank1>rank2:
        return True
    if rank1<rank2:
        return False
    if rank1 == rank2:
        if rank1==1:
            if value1>value2:
                return True
            else:
                return False
            
        if rank1==2:
            pass
        if rank1==3:
            pass
        if rank1 ==4:
            pass
        if rank1 ==5:
            pass
        if rank1==6:
            pass
        if rank1==7:
            the_four_c1, the_one_c1= find_numbers4(value1)
            the_four_c2, the_one_c2 = find_numbers4(value2)
        if rank1==8:
            c1 = max(value1)
            c2 = max(value2)
            if c1>c2:
                return True
            else:
                return False
            
        

    
    






def win_or_not_without_exchange(examples_input,K=0):
    type_and_values = []
    for i,v in enumerate(examples_input):

        values = get_rank_of_hand(hand=v)        
        type_and_values.append(values)
    
    
    remaning_cards = find_remaining_cards(examples_input)
        

    selectived_type_and_values = select_the_highest_rank_hands(type_and_values)
    
    # player 1 is not the lastest rank
    if selectived_type_and_values[0]!=type_and_values[0]:
        prob = 0
        print('%.4f' % prob)
        print(examples_input[0])
    
    # player 1 is the latest rank
    else:
        # player 1 is unqiue
        if len(selectived_type_and_values)==1:
            prob = 1
            print('%.7f' % prob)
            print(examples_input[0])
                
        
        # not unique, need discussion.
        else:
            highest_hands = get_the_higest_hand(selectived_type_and_values)
            if highest_hands[0] !=selectived_type_and_values[0]:
                if K =='0':
                    prob = 0
                    print('%.4f' % prob)
                    print(examples_input[0])
                else:
                    print(highest_hands[0])

            
            else:
                if len(highest_hands)>1:
                    if K=='0':
                        prob = 0
                        print('%.4f' % prob)
                        print(examples_input[0])
                    else:
                        pass
                else:
                    prob = 1
                    print('%.4f' % prob)
                    print(examples_input[0])
       

    
def find_remaining_cards(cards_list):
    # 定义所有花色和牌面值
    suits = ['S', 'H', 'D', 'C']  # Spades, Hearts, Diamonds, Clubs
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

    # 构建所有52张牌的集合
    full_deck = {f"{suit}{rank}" for suit in suits for rank in ranks}

    # 从输入列表中提取出所有已经出现的牌
    seen_cards = set()
    for hand in cards_list:
        seen_cards.update(hand.split())  # 添加当前手牌中的所有牌到已见牌集合中

    # 计算剩余的牌
    remaining_cards = list(full_deck - seen_cards)

    return remaining_cards


#  #### Input:

#   D8 H3 D9 S5 S3

#   C3 SQ C8 D7 D6

#   S4 C7 CJ C4 C5

#   DK DT C9 SK C6

#   H6 D4 DQ H7 H8

#   H4 HK HA C2 HQ

#   5

#   #### Your Output:

#   0.0000000

#   D8 H3 D9 S5 S3

#   #### Expected Output:

#   0.316017316017

#   ** H3 ** S5 S3

#   ** H3 D9 ** S3



if __name__=="__main__":


    
    examples_input = ["D8 H3 D9 S5 S3","C3 SQ C8 D7 D6","S4 C7 CJ C4 C5","DK DT C9 SK C6","H6 D4 DQ H7 H8","H4 HK HA C2 HQ"]
    
    

    win_or_not_without_exchange(examples_input,K='5')
