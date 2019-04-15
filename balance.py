import math
import traceback
import sys
import gym
import random
import csv
import numpy as np
import statistics as stats

LEFT_MOVE = 0
RIGHT_MOVE = 1

class QLearner:
    def __init__(self, alfa, gamma, eps, buckets, penalty, if_render):
        self.alfa = alfa
        self.gamma = gamma
        self.eps = eps
        self.buckets = buckets
        self.dict = {}
        self.default_reward = 0
        self.penalty = penalty
        self.render = if_render
        self.result = []

        self.environment = gym.make('CartPole-v1')
        self.attempt_no = 1
        self.upper_bounds = [
            self.environment.observation_space.high[0],
            0.5,
            self.environment.observation_space.high[2],
            math.radians(50)
        ]
        self.lower_bounds = [
            self.environment.observation_space.low[0],
            -0.5,
            self.environment.observation_space.low[2],
            -math.radians(50)
        ]

    def learn(self, max_attempts):
        for _ in range(max_attempts):
            reward_sum = self.attempt()
            self.result.append(reward_sum)
            # print(reward_sum)

    def attempt(self):
        observation = self.discretise(self.environment.reset())
        done = False
        reward_sum = 0.0
        while not done:
            if self.render:
                self.environment.render()
            action = self.pick_action(observation)
            new_observation, reward, done, info = self.environment.step(action)
            new_observation = self.discretise(new_observation)
            if done:
                reward = self.penalty
            self.update_knowledge(action, observation, new_observation, reward)
            observation = new_observation
            reward_sum += reward
        self.attempt_no += 1
        return reward_sum


    def get_discretised_bucket(self, observation_id, observation_value):
        min = self.lower_bounds[observation_id]
        max = self.upper_bounds[observation_id]
        range = max - min
        step_value = range/(self.buckets)
        step_current_range = min + step_value
        bucket_id = 0
        while step_current_range < max:
            if observation_value < step_current_range:
                return bucket_id
            bucket_id += 1
            step_current_range += step_value
        return bucket_id

    def discretise(self, observation):
        b0 = self.get_discretised_bucket(0,observation[0])
        b1 = self.get_discretised_bucket(1,observation[1])
        b2 = self.get_discretised_bucket(2,observation[2])
        b3 = self.get_discretised_bucket(3,observation[3])

        discretised = b0, b1, b2, b3
        return discretised

    def get_reward(self, observation, move):
        if (observation,move) in self.dict:
            return self.dict[(observation,move)]
        return self.default_reward

    def max_estimated_reward(self, observation):
        left_reward = self.get_reward(observation,LEFT_MOVE)
        right_reward = self.get_reward(observation,RIGHT_MOVE)
        return max(left_reward,right_reward)

    def pick_action(self, observation):
        if random.random() < self.eps:
            return self.environment.action_space.sample()
        left_reward = self.get_reward(observation,LEFT_MOVE)
        right_reward = self.get_reward(observation,RIGHT_MOVE)

        if left_reward > right_reward:
            return LEFT_MOVE
        elif right_reward > left_reward:
            return RIGHT_MOVE
        else:
            return self.environment.action_space.sample()

    def update_knowledge(self, action, observation, new_observation, reward):
        old_value = self.get_reward(observation,action)
        old_value_component = (1.0 - self.alfa) * old_value
        learned_value = reward + self.gamma * self.max_estimated_reward(new_observation)
        learned_value_component = self.alfa * learned_value
        new_value = old_value_component + learned_value_component
        self.dict[(observation,action)] = new_value

def main():
    argv = sys.argv
    attempts = 10000
    global_result = []
    try:
        alfa = float(argv[1])
        gamma = float(argv[2])
        eps = float(argv[3])
        buckets = int(argv[4])
        penalty = float(argv[5])
        if_render = True if argv[6] in ['True', 'true', '1'] else False
        repetitions = int(argv[7])
    except Exception:
        print('invalid or missing parameters: alfa gamma eps buckets penalty if_render repetitions')
        print(traceback.format_exc())
        sys.exit(1)

    for i in range(repetitions):
        learner = QLearner(alfa,gamma,eps,buckets,penalty,if_render)
        learner.learn(attempts)
        global_result.append(learner.result)

    file_name = "results_stddev/a{}_g{}_e{}_b{}.csv".format(alfa,gamma,eps,buckets)
    with open(file_name,'w') as f:
        writer = csv.writer(f,delimiter=',')
        writer.writerow(["attempt","mean","stddev"])
        for a in range(attempts):
            all_results_per_attempt = []
            for r in range(repetitions):
                all_results_per_attempt.append(global_result[r][a])

            mean = np.mean(all_results_per_attempt)
            if repetitions > 1:
                std = stats.stdev(all_results_per_attempt)
            else:
                std = 0.0
            writer.writerow([a+1,mean,std])
    f.close()

if __name__ == '__main__':
    main()
