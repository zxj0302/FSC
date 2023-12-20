import random
import math


def entropy(lst):
    freq = {}
    for item in lst:
        if item in freq:
            freq[str(item)] += 1
        else:
            freq[str(item)] = 1

    entropy = 0
    for count in freq.values():
        prob = count / len(lst)
        entropy -= prob * math.log2(prob)

    return entropy


for i in range(1, 50):
    data = [random.randint(1, 5) for _ in range(10)]
    exp = data + [random.randint(1, 5) for _ in range(1)]
    newint = [random.randint(1, 5)]

    a = entropy(data + newint) - entropy(data)
    b = entropy(exp + newint) - entropy(exp)

    if a - b < 0:
        print(str(a - b) + '=' + str(a) + '-' + str(b))
        print(str(data + newint))
        print(str(exp + newint))
