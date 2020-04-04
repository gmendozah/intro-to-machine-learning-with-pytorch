import numpy as np


# This function calculates entropy on two variables
def calculateTwoClassEntropy(n, m):
    return -(m / (m + n)) * np.log2(m / (m + n)) - (n / (m + n)) * np.log2(n / (m + n))


# This function calculates entropy on many variables
def calculateMultiClassEntropy(P):
    entropy = 0
    # p1 = m / (m + n)
    # p2 = n / (m + n)
    for i in range(len(P)):
        p_i = P[i] / sum(P)
        entropy += p_i * np.log2(p_i)
    return -1 * entropy


print(calculateTwoClassEntropy(4, 10))

print(calculateMultiClassEntropy([8, 3, 2]))
