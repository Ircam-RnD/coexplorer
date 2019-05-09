import copy
import numpy as np

class Env:

    def __init__(self, STATE_SIZE, STATE_STEPS, REWARD_LENGTH, REWARD):
        self.state_size = STATE_SIZE
        self.state_steps = STATE_STEPS
        self.reward_length = REWARD_LENGTH
        self.reward_size = REWARD

    def reset(self):
        state = np.reshape(np.ones(self.state_size)*0.5, [1,self.state_size])
        #state = np.reshape(np.ones(self.state_size) * 0., [1, self.state_size])
        return state

    def reset_random(self):
        states_inc = [i/float(self.state_steps) for i in range(self.state_steps+1)]
        state = np.reshape(np.random.choice(states_inc,self.state_size), [1,self.state_size])
        return state

    def step(self, state, action):

        new_state = copy.deepcopy(state[0])

        state_index = int(action / 2.0)
        new_stateValue = np.around(new_state[state_index] + 1.0/self.state_steps * (action % 2 * -2 + 1), decimals=5)
        new_stateValue = np.clip(new_stateValue,0,1)

        new_state[state_index] = new_stateValue
        new_state = np.reshape(new_state, [1,self.state_size])

        return new_state

    def set_reward(self,reward):
        rewards = np.zeros(self.reward_length)
        for i in range(len(rewards)):
            rewards[-i-1] = 0.9**i * reward
        return rewards