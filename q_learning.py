from environment import MountainCar
import sys
import numpy as np
import random


def main(args):
    # read inputs from command line
    mode = args[1]
    weight_out = args[2]
    returns_out = args[3]
    episodes = int(args[4])
    max_iterations = int(args[5])
    epsilon = float(args[6])
    gamma = float(args[7])
    learning_rate = float(args[8])
    # initialize the car, the weights, and the bias
    car = MountainCar(mode=mode)
    weights = np.zeros([car.action_space, car.state_space])
    bias = 0
    rewards = []
    # work through certain number of episodes
    for episode in range(episodes):
        # initialize current state and convert to sparse representation
        curr_state = car.reset()
        if mode == 'raw':
            curr_state = np.array(list(curr_state.values()))
        else:
            curr_state_keys = np.array(list(curr_state.keys()))
            curr_state = np.zeros(car.state_space)
            for i in curr_state_keys:
                curr_state[i] = 1.0
        # initialize number of iteration and total reward
        iteration = 0
        total_reward = 0
        # repeat until it reaches the maximum number of iterations
        while iteration < max_iterations:
            # calculate all q(s, a ; w)
            q_values = np.array([curr_state.dot(weights[action]) + bias for action in [0, 1, 2]])
            # choose the optimal action
            optimal_action = np.argmax(q_values)
            # select the optimal action w/ prob 1 - e and select uniformly w/ prob e
            if epsilon == 0.0:
                curr_action = optimal_action
            else:
                prob = random.uniform(0, 1)
                if prob > epsilon:
                    curr_action = optimal_action
                else:
                    curr_action = random.choice([0, 1, 2])
            # take the action and get the state (convert to sparse representation), reward, and the status
            next_state, reward, done = car.step(curr_action)
            if mode == 'raw':
                next_state = np.array(list(next_state.values()))
            else:
                next_state_keys = np.array(list(next_state.keys()))
                next_state = np.zeros(car.state_space)
                for i in next_state_keys:
                    next_state[i] = 1.0
            total_reward += reward
            # compute the gradient of the weights
            gradient_w = np.zeros([car.action_space, car.state_space])
            gradient_w[curr_action] = curr_state
            # compute all the q(s', a' ; w) based on s' (next_state)
            q_values = np.array([next_state.dot(weights[action]) + bias for action in [0, 1, 2]])
            # update weights and the bias term
            q_saw = curr_state.dot(weights[curr_action]) + bias
            weights = weights - learning_rate * (q_saw - (reward + gamma * np.amax(q_values))) * gradient_w
            bias = bias - learning_rate * (q_saw - (reward + gamma * np.amax(q_values)))
            # update state
            curr_state = next_state
            # increment the iteration
            iteration += 1
            # if the car reaches the destination, end this episode
            if done:
                break
        # store the total reward for this episode
        rewards.append(total_reward)

    # write the total reward to the output file
    with open(returns_out, 'w') as f:
        for r in rewards:
            f.write(str(r) + '\n')
    # write the bias term and the weights to the output file
    with open(weight_out, 'w') as f:
        f.write(str("%.16f" % bias) + '\n')
        # transpose the weights matrix to be |S| * |A|
        trans_weights = np.transpose(weights)
        for row in trans_weights:
            for entry in row:
                f.write(str("%.16f" % entry) + '\n')


if __name__ == "__main__":
    main(sys.argv)