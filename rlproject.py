
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt



# File paths
ratings_path = 'C:/Users/AW/Downloads/ml-1m/ratings.dat'
users_path = 'C:/Users/AW/Downloads/ml-1m/users.dat'
movies_path = 'C:/Users/AW/Downloads/ml-1m/movies.dat'

# Load data
ratings = pd.read_csv(ratings_path, sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
users = pd.read_csv(users_path, sep='::', engine='python', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
movies = pd.read_csv(movies_path, sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], encoding='ISO-8859-1')

# Merge ratings with users data to include Age
merged_data = ratings.merge(users[['UserID', 'Age']], on='UserID', how='left')

# Define age groups based on the Age value
age_groups = {
    1: 'Under 18',
    18: '18-24',
    25: '25-34',
    35: '35-44',
    45: '45-49',
    50: '50-55',
    56: '56+'
}

# Map age values to age group labels
merged_data = ratings.merge(users[['UserID', 'Age']], on='UserID', how='left')

# Define age groups based on the Age value
age_groups = {
    1: 'Under 18',
    18: '18-24',
    25: '25-34',
    35: '35-44',
    45: '45-49',
    50: '50-55',
    56: '56+'
}

# Map age values to age group labels
merged_data['AgeGroup'] = merged_data['Age'].map(age_groups)

# Compute average ratings for each age group
age_group_to_avg_rating = merged_data.groupby('AgeGroup')['Rating'].mean().reset_index()

# 排序以确保 "Under 18" 在第一位
age_group_to_avg_rating['AgeGroup'] = pd.Categorical(age_group_to_avg_rating['AgeGroup'], categories=list(age_groups.values()), ordered=True)
age_group_to_avg_rating.sort_values(by='AgeGroup', inplace=True)

# 获取平均评分
avg_ratings = age_group_to_avg_rating['Rating'].values

largest_avg_rating = age_group_to_avg_rating['Rating'].max()

# Set the number of arms (age groups)
age_group_labels = age_group_to_avg_rating['AgeGroup'].tolist()
n_arm = len(age_group_labels)

# Map age group labels to indices
age_group_to_index = {label: idx for idx, label in enumerate(age_group_labels)}

# Precompute ratings for each age group
age_group_ratings = {age: merged_data[merged_data['AgeGroup'] == age]['Rating'].values for age in age_groups.values()}
delta=3.766632-3.714512

# Print the DataFrame
print("Average ratings by age group:")
print(age_group_to_avg_rating)
print(largest_avg_rating)

"""Problem 1"""

# Get rewards for a given arm (age group)
def get_reward(arm):
    age_group_label = list(age_groups.values())[arm]
    ratings = age_group_ratings[age_group_label]
    if len(ratings) > 0:
        reward = random.choice(ratings)
    else:
        reward = 0
    return reward

# Compute regret
def regret(t, arm):
    best_arm_reward = largest_avg_rating  # Assuming largest_avg_rating is the average rating of the best arm
    if arm < len(avg_ratings):  # Ensure arm is within the index range
        chosen_arm_reward = avg_ratings[arm]
    else:
        chosen_arm_reward = 0  # Handle case where arm index is out of range
    return best_arm_reward - chosen_arm_reward
print(avg_ratings)

average_rewards = 0
for i in range(0,10000):
    reward=get_reward(6)
    average_rewards=(average_rewards*i+reward)/(i+1)
print(average_rewards)

# Define constants
n_round = 50000
n_experiments = 10
num_total_experiments = 100
exploration_round = n_round//10  # 10% of n_round

# Number of arms should match the number of age groups
assert n_arm == len(avg_ratings), "Number of arms should match the number of age groups"

# Define the ETC algorithm
def ETC_algorithm(n_round, n_arm, exploration_round):
    cumulative_regret = np.zeros(n_round)
    arm_counts = np.zeros(n_arm)
    arm_rewards = np.zeros(n_arm)

    arm = 0
    reward = get_reward(arm)
    arm_rewards[arm] += reward
    arm_counts[arm] += 1
    cumulative_regret[0] = regret(0, 0)

    for t in range(1,n_round):
        if t < exploration_round:
            arm = t % n_arm

            cumulative_regret[t] = cumulative_regret[t - 1] + regret(t, arm)
        else:
            arm = np.argmax(arm_rewards / arm_counts)  # Exploit best arm
            cumulative_regret[t] = cumulative_regret[t - 1] + regret(t, arm)
        reward = get_reward(arm)
        arm_rewards[arm] += reward
        arm_counts[arm] += 1

    return cumulative_regret


# Define the UCB algorithm
def UCB_algorithm(n_round, n_arm, B=4):
    cumulative_regret = np.zeros(n_round)
    arm_counts = np.zeros(n_arm)
    arm_rewards = np.zeros(n_arm)

    for t in range(n_arm):
        reward = get_reward(t)
        arm_rewards[t] += reward
        arm_counts[t] += 1
        cumulative_regret[t] = regret(t, t) + cumulative_regret[t - 1]

    for t in range(n_arm, n_round):
        ucb_values = arm_rewards / arm_counts + B * np.sqrt(np.log(n_round) / arm_counts)
        arm = np.argmax(ucb_values)
        reward = get_reward(arm)
        arm_rewards[arm] += reward
        arm_counts[arm] += 1
        cumulative_regret[t] = regret(t, arm) + cumulative_regret[t - 1]

    return cumulative_regret

# Define the TS algorithm
def TS_algorithm(n_round, n_arm, B=4):
    cumulative_regret = np.zeros(n_round)
    arm_counts = np.ones(n_arm)
    arm_rewards = np.zeros(n_arm)

    for t in range(n_arm):
        reward = get_reward(t)
        arm_rewards[t] += reward
        cumulative_regret[t] = regret(t, t)

    for t in range(n_arm, n_round):
        theta = np.random.normal(arm_rewards / arm_counts, B / (2 * np.sqrt(arm_counts)))
        arm = np.argmax(theta)
        reward = get_reward(arm)
        arm_rewards[arm] += reward
        arm_counts[arm] += 1
        cumulative_regret[t] = regret(t, arm) + cumulative_regret[t - 1]

    return cumulative_regret

# Function to plot average regret with error bars
def plot_avg_with_error_bars(all_regrets, title):
    mean_regret = np.mean(all_regrets, axis=0)
    std_dev = np.std(all_regrets, axis=0)
    plt.plot(mean_regret, label='Average Regret')
    plt.fill_between(range(len(mean_regret)), mean_regret - std_dev, mean_regret + std_dev, alpha=0.5, label='1 Std Dev')
    plt.title(title)
    plt.xlabel('Round')
    plt.ylabel('Cumulative Regret')
    plt.legend()

    plt.show()

# Function to run experiments
def run_experiments(algorithm, n_experiments, n_round, n_arm, *args):
    all_regrets = []
    for _ in range(n_experiments):
        cumulative_regret = algorithm(n_round, n_arm, *args)
        all_regrets.append(cumulative_regret)
    return np.array(all_regrets)

# Function to plot results
def plot_results(all_regrets, title):
    for experiment in all_regrets:
        plt.plot(experiment, alpha=0.5)
    plt.title(title)
    plt.xlabel('Round')
    plt.ylabel('Cumulative Regret')

    plt.show()



exploration_round = 5000
etc_regrets = run_experiments(ETC_algorithm, n_experiments, n_round, n_arm, exploration_round)
plot_results(etc_regrets, 'ETC Algorithm')


ucb_regrets = run_experiments(UCB_algorithm, n_experiments, n_round, n_arm)
plot_results(ucb_regrets, 'UCB Algorithm')

ts_regrets = run_experiments(TS_algorithm, n_experiments, n_round, n_arm)
plot_results(ts_regrets, 'TS Algorithm')

# Run 100 experiments and plot the average with error bars
n_experiments = 100

etc_regrets = run_experiments(ETC_algorithm, n_experiments, n_round, n_arm, 5000)
plot_avg_with_error_bars(etc_regrets, 'ETC Algorithm')

ucb_regrets = run_experiments(UCB_algorithm, n_experiments, n_round, n_arm, 4)
plot_avg_with_error_bars(ucb_regrets, 'UCB Algorithm')

ts_regrets = run_experiments(TS_algorithm, n_experiments, n_round, n_arm, 4)
plot_avg_with_error_bars(ts_regrets, 'TS Algorithm')

"""ETC had the largest change in each experiment, and the other two changed but only minimally.
 So the variance of ETC should be the largest. eTC occasionally has a line that deviates from the others.
This could be because the wrong ARM was selected."""

"""Problem 2"""

# Function to plot results
def plot_results_for_n(n_values, n_experiments, n_arm, B):
    for n in n_values:
        exploration_round = int(0.1 * n)

        etc_regrets = run_experiments(ETC_algorithm, n_experiments, n, n_arm, exploration_round)
        ucb_regrets = run_experiments(UCB_algorithm, n_experiments, n, n_arm, B)
        ts_regrets = run_experiments(TS_algorithm, n_experiments, n, n_arm, B)

        etc_mean_regret = np.mean(etc_regrets, axis=0)
        etc_std_dev = np.std(etc_regrets, axis=0)

        ucb_mean_regret = np.mean(ucb_regrets, axis=0)
        ucb_std_dev = np.std(ucb_regrets, axis=0)

        ts_mean_regret = np.mean(ts_regrets, axis=0)
        ts_std_dev = np.std(ts_regrets, axis=0)

        plt.figure(figsize=(12, 7))

        # Plotting ETC with error bars
        plt.plot(etc_mean_regret, label='ETC',color='blue')
        plt.fill_between(range(n), etc_mean_regret - etc_std_dev, etc_mean_regret + etc_std_dev, color='blue',alpha=0.2)

        # Plotting UCB with error bars
        plt.plot(ucb_mean_regret, label='UCB',color='green')
        plt.fill_between(range(n), ucb_mean_regret - ucb_std_dev, ucb_mean_regret + ucb_std_dev, color='green',alpha=0.2)

        # Plotting TS with error bars
        plt.plot(ts_mean_regret, label='TS',color='red')
        plt.fill_between(range(n), ts_mean_regret - ts_std_dev, ts_mean_regret + ts_std_dev, color='red', alpha=0.2)

        plt.title(f'Average Cumulative Regret for n = {n}')
        plt.xlabel('Round')
        plt.ylabel('Cumulative Regret')
        plt.legend()
        plt.savefig('C:/Users/AW/Downloads/graph/21')
        plt.show()


n_values = [500, 5000, 50000, 500000]
n_experiments = 100
B = 4

plot_results_for_n(n_values, n_experiments, n_arm, B)

n_experiments=100
n_round=5000000
exploration_round=500000
etc_regrets = run_experiments(ETC_algorithm, n_experiments, n_round, n_arm, exploration_round)


ucb_regrets = run_experiments(UCB_algorithm, n_experiments, n_round, n_arm, B)
ts_regrets = run_experiments(TS_algorithm, n_experiments, n_round, n_arm, B)
etc_mean_regret = np.mean(etc_regrets, axis=0)
etc_std_dev = np.std(etc_regrets, axis=0)

ucb_mean_regret = np.mean(ucb_regrets, axis=0)
ucb_std_dev = np.std(ucb_regrets, axis=0)

ts_mean_regret = np.mean(ts_regrets, axis=0)
ts_std_dev = np.std(ts_regrets, axis=0)

n = len(etc_mean_regret)  # Assuming all have the same length

plt.figure(figsize=(10, 6))

# Plotting ETC with error bars
plt.plot(etc_mean_regret, label='ETC', color='blue')
plt.fill_between(range(n), etc_mean_regret - etc_std_dev, etc_mean_regret + etc_std_dev, color='blue', alpha=0.2)

# Plotting UCB with error bars
plt.plot(ucb_mean_regret, label='UCB', color='green')
plt.fill_between(range(n), ucb_mean_regret - ucb_std_dev, ucb_mean_regret + ucb_std_dev, color='green', alpha=0.2)

# Plotting TS with error bars
plt.plot(ts_mean_regret, label='TS', color='red')
plt.fill_between(range(n), ts_mean_regret - ts_std_dev, ts_mean_regret + ts_std_dev, color='red', alpha=0.2)

plt.title('Average Cumulative Regret for n=5000000')
plt.xlabel('Round')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.show()

"""
TS starts at n=50000, it starts to have a tendency to show logarithmic regret behavior.
And at n=500000, both TS and UCB show logarithmic regret behavior.
So I think TS will show logarithmic regret behavior earlier,
or maybe it and UCB are showing logarithmic regret behavior at the same time.
When n is small, the performance is similar. Although n becomes larger, the advantage of TS algorithm becomes more and more obvious.
"""

"""Problem 3"""

n_experiments=100
n_round=50000
n_arm=7
etc_regrets = run_experiments(ETC_algorithm, n_experiments, n_round, n_arm, 50)
plot_avg_with_error_bars(etc_regrets, 'ETC Algorithm')
etc_regrets = run_experiments(ETC_algorithm, n_experiments, n_round, n_arm, 500)
plot_avg_with_error_bars(etc_regrets, 'ETC Algorithm')
etc_regrets = run_experiments(ETC_algorithm, n_experiments, n_round, n_arm, 2000)
plot_avg_with_error_bars(etc_regrets, 'ETC Algorithm')
etc_regrets = run_experiments(ETC_algorithm, n_experiments, n_round, n_arm, 5000)
plot_avg_with_error_bars(etc_regrets, 'ETC Algorithm')
etc_regrets = run_experiments(ETC_algorithm, n_experiments, n_round, n_arm, 10000)
plot_avg_with_error_bars(etc_regrets, 'ETC Algorithm')


"""
ETC performs better when mk equals 2000 or 5000. At this point ETC is better than UCB but not as good as TS
"""

"""Problem 4"""
def UCB_l_algorithm(n_round, n_arm, l, B=4):
    cumulative_regret = np.zeros(n_round)
    arm_counts = np.zeros(n_arm)
    arm_rewards = np.zeros(n_arm)

    for t in range(n_arm):
        reward = get_reward(t)
        arm_rewards[t] += reward
        arm_counts[t] += 1
        cumulative_regret[t] = regret(t, t) + cumulative_regret[t - 1]

    for t in range(n_arm, n_round):
        ucb_values = arm_rewards / arm_counts + B/2 * np.sqrt(l*np.log(n_round) / arm_counts)
        arm = np.argmax(ucb_values)
        reward = get_reward(arm)
        arm_rewards[arm] += reward
        arm_counts[arm] += 1
        cumulative_regret[t] = regret(t, arm) + cumulative_regret[t - 1]

    return cumulative_regret

def AS_UCB_algorithm(n_round, n_arm):
    cumulative_regret = np.zeros(n_round)
    arm_counts = np.zeros(n_arm)
    arm_rewards = np.zeros(n_arm)

    for t in range(n_arm):
        reward = get_reward(t)
        arm_rewards[t] += reward
        arm_counts[t] += 1
        cumulative_regret[t] = regret(t, t)

    for t in range(n_arm, n_round):
        ucb_values = arm_rewards / arm_counts + 2 * np.sqrt(2*np.log(1+t*np.log(t)**2) / arm_counts)
        arm = np.argmax(ucb_values)
        reward = get_reward(arm)
        arm_rewards[arm] += reward
        arm_counts[arm] += 1
        cumulative_regret[t] = regret(t, arm) + cumulative_regret[t - 1]

    return cumulative_regret

as_ucb_regrets = run_experiments(AS_UCB_algorithm, n_experiments, n_round, n_arm)

plot_avg_with_error_bars(as_ucb_regrets, 'Asymptotical UCB Algorithm')

ucb_regrets = run_experiments(UCB_l_algorithm, n_experiments, n_round, n_arm, 1, 4)

plot_avg_with_error_bars(ucb_regrets, 'UCB Algorithm, l=1')

ucb_regrets = run_experiments(UCB_l_algorithm, n_experiments, n_round, n_arm, 2, 4)

plot_avg_with_error_bars(ucb_regrets, 'UCB Algorithm, l=2')

ucb_regrets = run_experiments(UCB_l_algorithm, n_experiments, n_round, n_arm, 4, 4)

plot_avg_with_error_bars(ucb_regrets, 'UCB Algorithm, l=4')


"""
#l=4 has the worst performance, then asymptotic UCB, then l=2, and the best performance when l=1.

"""
"""Problem 5"""
n_experiments=100
n_round=1000000
exploration_round=100000
B=4
etc_regrets = run_experiments(ETC_algorithm, n_experiments, n_round, n_arm, exploration_round)

ucb_regrets = run_experiments(UCB_algorithm, n_experiments, n_round, n_arm, B)

ts_regrets = run_experiments(TS_algorithm, n_experiments, n_round, n_arm, B)

as_ucb_regrets = run_experiments(AS_UCB_algorithm, n_experiments, n_round, n_arm)

# Calculate mean and standard deviation for each algorithm's regrets
etc_mean_regret = np.mean(etc_regrets, axis=0)
etc_std_dev = np.std(etc_regrets, axis=0)

ucb_mean_regret = np.mean(ucb_regrets, axis=0)
ucb_std_dev = np.std(ucb_regrets, axis=0)

ts_mean_regret = np.mean(ts_regrets, axis=0)
ts_std_dev = np.std(ts_regrets, axis=0)

as_ucb_mean_regret = np.mean(as_ucb_regrets, axis=0)
as_ucb_std_dev = np.std(as_ucb_regrets, axis=0)

n = len(etc_mean_regret)  # Assuming all have the same length

plt.figure(figsize=(12, 7))

# Plotting ETC with error bars
plt.plot(etc_mean_regret, label='ETC', color='blue')
plt.fill_between(range(n), etc_mean_regret - etc_std_dev, etc_mean_regret + etc_std_dev, color='red', alpha=0.2)

# Plotting UCB with error bars

plt.plot(ucb_mean_regret, label='UCB', color='green')
plt.fill_between(range(n), ucb_mean_regret - ucb_std_dev, ucb_mean_regret + ucb_std_dev, color='blue', alpha=0.2)

# Plotting TS with error bars

plt.plot(ts_mean_regret, label='TS', color='red')
plt.fill_between(range(n), ts_mean_regret - ts_std_dev, ts_mean_regret + ts_std_dev, color='orange', alpha=0.2)

# Plotting AS_UCB with error bars

plt.plot(as_ucb_mean_regret, label='AS_UCB', color='purple')
plt.fill_between(range(n), as_ucb_mean_regret - as_ucb_std_dev, as_ucb_mean_regret + as_ucb_std_dev, color='green', alpha=0.2)

plt.title('Average Cumulative Regret')
plt.xlabel('Round')
plt.ylabel('Cumulative Regret')
plt.legend()

plt.show()
