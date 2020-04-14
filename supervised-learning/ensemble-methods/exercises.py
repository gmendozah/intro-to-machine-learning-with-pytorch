import numpy as np


def weight(correct, incorrect):
    return np.log(correct / incorrect)


print(weight(7, 1))

print(weight(4, 4))

print(weight(2, 6))
