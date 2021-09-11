"""
Upper Confidence Bound (UCB)
"""
# Importing the libraries
from math import sqrt, log
from pandas import read_csv
from matplotlib.pyplot import hist, title, xlabel, ylabel, show

# Importing the dataset
dataset = read_csv("Ads_CTR_Optimisation.csv")
rewards = read_csv("Ads_CTR_Optimisation.csv").values

# Implementing UCB
N = len(dataset)
D = len(dataset.columns)
ads_selected = []
numbers_of_selections = [0] * D
sums_of_rewards = [0] * D
TOTAL_REWARDS = 0
for n in range(N):
    AD = 0
    MAX_UPPER_BOUND = 0
    for i in range(D):
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = sqrt(1.5 * log(n + 1) / numbers_of_selections[i])
            UPPER_BOUND = average_reward + delta_i
        else:
            UPPER_BOUND = 1e400
        if UPPER_BOUND > MAX_UPPER_BOUND:
            MAX_UPPER_BOUND = UPPER_BOUND
            AD = i
    ads_selected.append(AD)
    numbers_of_selections[AD] += 1
    reward = rewards[n, AD]
    sums_of_rewards[AD] += reward
    TOTAL_REWARDS += reward

# Visualising the results - Histogram
hist(ads_selected)
title("Histogram of ads selections")
xlabel("Ads")
ylabel("Number of times each ad was selected")
show()
