import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import sys

Accuracy = []
Cost = []
Offloaded = []
num_exits = 7
n_epoch = 3060
for k in range(5):
    df = pd.read_csv('/home/divya/Aastha_early_exits_work/AdaEE/confidences/E18_Ptest.csv')
    df = df.sample(frac=1)
    true_labels = df.iloc[:, -1]
    df = df.to_numpy()
    conf_list = df[:, 1::2]
    class_list = df[:, 2::2]
    num_actions = np.zeros(num_exits)
    total_rewards = np.zeros(num_exits)
    # selected_exit = np.random.randint(0, 7)

    def reward_calculator(conf_list, class_list, alpha, n_exit, mu, o):
        reward, cost, offload = None, None, False
    
        confidence = conf_list[n_exit] if n_exit < 6 else conf_list[6]
        gamma = 1/10
        if confidence>=alpha and n_exit<=5:
            cost = gamma * (n_exit+1+(2*(n_exit + 1)))
            reward = confidence - mu * gamma * (n_exit+1+(2*(n_exit + 1)))
            pred = class_list[n_exit]
            count = 1
        elif n_exit<=5 and confidence<alpha:
            cost = o + gamma * (n_exit+1+(2*(n_exit + 1)))
            reward = conf_list[6] - mu * o - mu * gamma * (n_exit+1+(2*(n_exit + 1)))
            pred = class_list[6]
            count = 0
            offload = True
        elif n_exit==6:
            cost = gamma * 19
            reward = confidence - mu * gamma * 19
            pred = class_list[6]
            count = 1
        return reward, cost, offload, pred, count

    def accuracy_generate(y,y_pred):
        return np.sum(np.array(y_pred)==y)/len(y)



    # Initialize variables to track the number of times each action is selected and the total rewards for each action


    # Main UCB loop
    r_optimal = -69059.76
    o = 0.5
    t = 7
    c = 1
    mu = 1
    alpha = 0.5
    count_1 = 0
    total_cost = 0
    predictions = []
    selected_exit_set = []
    times_selected = []
    sample_offloaded = []
    # Initialize by playing each arm once
    for arm in range(num_exits):
        random_sample=np.random.choice(range(3060))
        reward, _, _, _, _ = reward_calculator(conf_list[random_sample], class_list[random_sample], alpha, arm, mu, o)
        num_actions[arm] += 1
        total_rewards[arm] += reward
    for step in range(n_epoch):
        offload_flag = 0
        ucb_values = [
            total_rewards[exit_num] / max(1, num_actions[exit_num]) +
            c * np.sqrt(np.log(t + 1) / max(1, num_actions[exit_num]))
            for exit_num in range(num_exits)
        ]
        # Choose the action with the highest UCB value
        selected_exits = []
        selected_exit = np.argmax(ucb_values)
        # selected_exit = np.random.randint(0, 7)
        for i in range(selected_exit):
            selected_exits.append(i)
            selected_exits.append(selected_exit)
            
        # Get the reward for the selected exit
        reward, cost, offload, prediction, count = reward_calculator(conf_list[step], class_list[step], alpha, selected_exit, mu, o)
        total_cost+=cost
        
        for selected_exit in selected_exits:
            reward_1, _, _, _, _ = reward_calculator(conf_list[step], class_list[step], alpha, selected_exit, mu, o)
            num_actions[selected_exit] += 1
            total_rewards[selected_exit] += reward_1
    
        if count == 1:
            count_1+=1
            selected_exit_set.append(selected_exit)
        else:
            offload_flag+=offload
            sample_offloaded.append(offload_flag)
        predictions.append(prediction)

    # Update the number of times the selected exit is chosen and the total rewards
    # selected_exit_set.append(selected_exit)
    
        for i in selected_exit_set:
            num_actions[i] += 1
            total_rewards[i] += reward
        
        t += 1

    for i in range(num_exits):
        times_selected.append(selected_exit_set.count(i))

    final_exit_set = [0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,1]
    j=0
    while j<=5:
        for i in range(len(final_exit_set)):
            if final_exit_set[i]==1:
                final_exit_set[i]=times_selected[j]
                j+=1  
            
    # for i in range(1, 20):
    #     cost_2_nr+=final_exit_set[i-1]*i
    #     cost_2_dr+=final_exit_set[i-1]*19
    Accuracy.append(accuracy_generate(true_labels, predictions)*100)
    Cost.append(total_cost)
    Offloaded.append(len(sample_offloaded))

    # print('Number of offloaded samples:', len(sample_offloaded))
    # print('Exit selection count:', times_selected)
    print('Accuracy:', accuracy_generate(true_labels, predictions)*100)
    # print('Total Cost:', total_cost)
    # print('Total Cost 2:', 1-(cost_2_nr/cost_2_dr))
    # print(final_exit_set)

print(sum(Accuracy)/5)
print(sum(Cost)/5)
# print(sum(Offloaded)/5)
    
