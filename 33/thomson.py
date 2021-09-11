"""
Thompson Sampling
"""
# Importing the libraries
from random import betavariate
from pandas import read_csv
from matplotlib.pyplot import hist, title, xlabel, ylabel, show

# Importing the dataset
dataset = read_csv("Ads_CTR_Optimisation.csv")
rewards = read_csv("Ads_CTR_Optimisation.csv").values

# Implementing Thomson Sampling
N = len(dataset)
D = len(dataset.columns)
ads_selected = []
numbers_of_rewards_1 = [0] * D
numbers_of_rewards_0 = [0] * D
TOTAL_REWARD = 0
for n in range(N):
    AD = 0
    MAX_RANDOM = 0
    for i in range(D):
        random_beta = betavariate(
            numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1
        )
        if random_beta > MAX_RANDOM:
            MAX_RANDOM = random_beta
            AD = i
    ads_selected.append(AD)
    reward = rewards[n, AD]
    if reward == 1:
        numbers_of_rewards_1[AD] += 1
    else:
        numbers_of_rewards_0[AD] += 1
    TOTAL_REWARD += reward

# Visualising the results - Histogram
hist(ads_selected)
title("Histogram of ads selections")
xlabel("Ads")
ylabel("Number of times each ad was selected")
show()
