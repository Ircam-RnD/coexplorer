#############################################################################################################
# TODO: - Create separate thread for training so state transition time is exact
# TODO: - Duplicate code for exploring_starts and random_action functions (pseudo-count/prediction gain calculation)
#############################################################################################################
import os
import sys
import time
import copy
import pickle
import random
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from agent import DTAMERAgent, Memory
from tracker import Tracker
from environment import Env
from OSCinterface import OSCClass

# TRAINING_PARAMS = [STATES, STEPS, HL_NB, HL_SIZE, EPS_DEC, LR, REWARD_LEN, REWARD, REPLAY_SIZE, BATCH_SIZE, EPS_START]
# TRAINING_PARAMS_1 = [16, 1, 2, 100, 0, 0.001, 4, 1, 0, 4, 0.5]
TRAINING_PARAMS_1 = [10, 100, 2, 100, 2000, 0.002, 10, 1, 700, 32, 0.1]
# TRAINING_PARAMS_1 = [1, 100, 2, 100, 2000, 0.002, 10, 1, 700, 32, 0.1]

#TRAINING_LABEL = 'TEST'
TRAINING_LABEL = input('Enter NAME_ITERATION (e.g. ALEX_2):')
TRANSITION_TIME = .1
MAX_TRANSITION_TIME = 1

MAX_REWARD_LENGTH = 64
MAX_STATE_STEPS = 100
PRINT_FREQ = 250

TRAINING = TRAINING_PARAMS_1
STATE_SIZE = TRAINING[0]
ACTION_SIZE = 2 * STATE_SIZE
STATE_STEPS = TRAINING[1]
HIDDEN_LAYER_NB = TRAINING[2]
HIDDEN_LAYER_SIZE = TRAINING[3]
EPS_DECAY = TRAINING[4]
LEARNING_RATE = TRAINING[5]
REWARD_LENGTH = TRAINING[6]
REWARD = TRAINING[7]
REPLAY_SIZE = TRAINING[8]
BATCH_SIZE = TRAINING[9]
EPS_START = TRAINING[10]

def init_program(started_bool = False):
    global save_path

    tf.reset_default_graph()
    sess = tf.Session()

    agent = DTAMERAgent(STATE_SIZE, ACTION_SIZE, HIDDEN_LAYER_NB, HIDDEN_LAYER_SIZE, EPS_DECAY, LEARNING_RATE, REWARD_LENGTH, REWARD, TRANSITION_TIME, REPLAY_SIZE, BATCH_SIZE, EPS_START)
    env = Env(STATE_SIZE, STATE_STEPS, REWARD_LENGTH, REWARD)
    tracker = Tracker(STATE_SIZE, MAX_STATE_STEPS, TRAINING_LABEL)

    if not started_bool and os.path.isdir('./datalogs/' + TRAINING_LABEL):
        print('Choose a new name for this training run')
        sys.exit()

    if not os.path.isdir('./datalogs/'  + TRAINING_LABEL):
        os.makedirs('./datalogs/'  + TRAINING_LABEL)

    sess.run(tf.global_variables_initializer())
    save_path = r'./datalogs/' + TRAINING_LABEL

    print('State steps|increment = ' + str(env.state_steps) + '|' + str(1.0 / env.state_steps))

    return sess, agent, env, tracker

def resample_actions(env, t, resample_factor):


    state_steps = int(max(2,min(env.state_steps * resample_factor,MAX_STATE_STEPS)))
    env.state_steps = state_steps

    print('time; ' + str(t) + '; Resample! Increment = ' + str(1.0 / state_steps))

def adjust_reward_length(agent, t, reward_length_factor):

    #new_reward_length = int(max(1,min(agent.reward_length * reward_length_factor,MAX_REWARD_LENGTH)))
    agent.reward_length = int(max(1,min(agent.reward_length + reward_length_factor,MAX_REWARD_LENGTH)))
    env.reward_length = agent.reward_length

    temp_memory = copy.deepcopy(agent.reward_memory)
    temp_memory2 = copy.deepcopy(agent.delay_memory)

    agent.reward_memory = Memory(agent.reward_length, agent.state_size)
    agent.delay_memory = Memory(int(np.ceil(0.2 / agent.transition_time + agent.reward_length)),
                                          agent.state_size)

    agent.reward_memory.buffer.extend(temp_memory.buffer)
    agent.delay_memory.buffer.extend(temp_memory2.buffer)

    print('time; ' + str(t) + '; New reward length! Reward length = ' + str(agent.reward_length))

def rescale_transitions(agent, t):
    global TRANSITION_TIME

    #TRANSITION_TIME = max(0.015625,min(TRANSITION_TIME * trans_time,MAX_TRANSITION_TIME))
    TRANSITION_TIME = 1.0/agent.reward_length
    agent.transition_time = TRANSITION_TIME

    temp_memory2 = copy.deepcopy(agent.delay_memory)
    agent.delay_memory = Memory(int(np.ceil(0.2 / agent.transition_time + agent.reward_length)),
                                          agent.state_size)
    agent.delay_memory.buffer.extend(temp_memory2.buffer)

    print('time;' + str(t) + '; New transition time! Transition time = ' + str(TRANSITION_TIME))

def explore_state(sess, agent, env, tracker, t, interfaceMax):

    state = 0
    prediction_gain = -10
    next_density = copy.deepcopy(agent.density_weights)
    for i in range(agent.state_size*4):
        state_nxt = env.reset_random()
        tiles_idx = agent.calc_tiles_idx(state_nxt[0])
        state_prob_nxt = np.sum(agent.density_weights[tiles_idx])/((t+1) * agent.numtilings)

        next_density[tiles_idx] += 1
        next_state_prob_nxt = np.sum(next_density[tiles_idx]) / ((t+2) * agent.numtilings)
        next_density[tiles_idx] -= 1

        prediction_gain_nxt = np.log(next_state_prob_nxt) - np.log(state_prob_nxt)

        if prediction_gain_nxt > prediction_gain:
            prediction_gain = copy.deepcopy(prediction_gain_nxt)
            state = copy.deepcopy(state_nxt)

    print('time; ' + str(t) + '; Explore from new state! : ' + str(state))
    tracker.fill_trajectory(state,'Explore_state')
    interfaceMax.send_state_to_slider(state, 'Explore_state')
    action, rand_bool = agent.act(sess, state)

    # timeout_start = time.time()
    # reward_idx = 1

    interfaceMax.client.send_message('/params', state[0])

    # Following code added to assure reward is distributed over appropriate reward_length (ex. When explore_state
    # and then assigning reward, needs to be distributed from explore_state onwards -> variable reward_length size)
    # reward = 0
    # if not interfaceMax.paused:
    #     while time.time() < (timeout_start + (agent.reward_length * TRANSITION_TIME)):
    #         next_state = env.step(state, action)
    #         next_action, rand_bool = agent.act(sess, next_state, t)
    #         agent.remember_transition(state, action)
    #         interfaceMax.client.send_message('/params', state[0])
    #
    #
    #         state = next_state
    #         action = next_action
    #
    #         while time.time() < (timeout_start + (reward_idx * TRANSITION_TIME)):
    #             reward = interfaceMax.reward
    #             interfaceMax.client.send_message('/reward_in', reward)
    #
    #         interfaceMax.send_state_to_slider(state, reward)
    #
    #         reward = 0
    #         interfaceMax.reward = 0
    #         reward_idx += 1
    #         t += 1
    #
    #     reward = interfaceMax.reward
    #     interfaceMax.reward = 0
    #     interfaceMax.received = False
    #     rewards = env.set_reward(reward)
    #
    #     if not reward == 0:
    #         tracker.fill_trajectory(state, reward)
    #         agent.remember_rewards(rewards)
    return state, action, t

def explore_action(agent, state, t):
    action = 999
    prediction_gain = -10
    #next_density = copy.deepcopy(agent.density_weights)
    invalid_actions = [ind * 2 + 1 if x == 0 else ind * 2 for ind, x in enumerate(state[0]) if x in (0, 1)]

    for i in range(agent.state_size * 2):
        test_state = env.step(state, i)
        tiles_idx = agent.calc_tiles_idx(test_state[0])
        test_state_prob = np.sum(agent.density_weights[tiles_idx]) / ((t + 1) * agent.numtilings + 1)

        agent.density_weights[tiles_idx] += 1
        test_state_prob_nxt = np.sum(agent.density_weights[tiles_idx]) / ((t + 2) * agent.numtilings + 1)
        agent.density_weights[tiles_idx] -= 1

        prediction_gain_nxt = np.log(test_state_prob_nxt) - np.log(test_state_prob)

        if prediction_gain_nxt > prediction_gain and i not in invalid_actions:
            prediction_gain = copy.deepcopy(prediction_gain_nxt)
            action = i

    print('time; ' + str(t) + '; Explore new states! : ' + str(state))

    return action

def explore_random_action(agent, state, t):
    action = 999
    invalid_actions = [ind * 2 + 1 if x == 0 else ind * 2 for ind, x in enumerate(state[0]) if x in (0, 1)]

    i = random.choice(range(agent.state_size * 2))

    while i in invalid_actions:
        i = random.choice(range(agent.state_size * 2))
    else:
        action = i

    print('time; ' + str(t) + '; Explore random action! : ' + str(state))

    return action

def super_like(agent, env, tracker, state, score):

    if interfaceMax.paused:
        start_state = copy.deepcopy(state)
        print(start_state)
    else:
        start_state = copy.deepcopy(agent.delay_memory.sample(1))[0][0]

    super_like_size = agent.reward_length

    temp_state = copy.deepcopy(start_state)

    for i in range(agent.state_size):
        action = i * 2
        for j in range(2):
            for k in range(max(1,int(super_like_size/2))):
                if not (temp_state[0][i] == 1 and ((action + j) % 2) == 0) and not (temp_state[0][i] == 0 and
                                                                                    ((action + j) % 2) == 1):
                    next_state = env.step(temp_state, action + j)
                    agent.reward_memory.add(np.array([(temp_state, action + j, - 2 * score * agent.reward_size)]))
                    agent.reward_memory.add(np.array([(next_state, action - j + 1, 2 * score * agent.reward_size)]))
                    temp_state = next_state
            if len(agent.reward_memory.buffer) == agent.reward_length: # Bugfix: To prevent superlike after
                                                                # adjust_reward_length with state[0]=1
                batch = np.reshape(agent.reward_memory.buffer, [agent.reward_length, 3])
                agent.train(sess, batch)
            temp_state = start_state
            agent.replay_memory.add(agent.reward_memory.buffer)

    if score == 1:
        print('!! SUPERLIKE !! for ' + str(start_state))
        tracker.fill_trajectory(start_state, 'Superlike')
        interfaceMax.send_state_to_slider(state, 'Superlike')
    elif score == -1:
        print('!! SUPERDISLIKE !! for ' + str(start_state))
        tracker.fill_trajectory(start_state, 'Superdislike')
        interfaceMax.send_state_to_slider(state, 'Superdislike')

if __name__ == "__main__":

    # Init classes
    sess, agent, env, tracker = init_program(started_bool = False)
    interfaceMax = OSCClass(STATE_SIZE, ACTION_SIZE, TRANSITION_TIME, "127.0.0.1", 5005, TRAINING_LABEL)

    # Init state, action and variables
    reward = 0
    t_idx = 0
    nb_iter = 0
    rewards = np.zeros(agent.reward_length)
    state = env.reset()
    action, rand_bool = agent.act(sess, state)

    # First loop, wait here until user starts interaction
    interfaceMax.client.send_message('/path', str(os.getcwd()) + '/datalogs/' + ' ' + TRAINING_LABEL)
    interfaceMax.send_workflow_control(init=1)

    while interfaceMax.paused and interfaceMax.running:
        time.sleep(0.01)

        if interfaceMax.load:
            agent.load_model(sess, interfaceMax.load_modelname)
            interfaceMax.load = False
    interfaceMax.send_workflow_control(paused = 0)
    interfaceMax.VSTsample_bool = False

    # Outer loop
    while interfaceMax.running:
        ########################################################
        ##############      RL CYCLE         ###################
        ########################################################

        next_state = env.step(state, action)
        next_action, rand_bool = agent.act(sess, next_state, t_idx)
        agent.remember_transition(state, action)

        interfaceMax.send_state(next_state[0])
        interfaceMax.send_workflow_control(rand = rand_bool)

        # Inner loop, get reward during or after transition
        timeout_start = time.time()
        while time.time() < (timeout_start + TRANSITION_TIME):
            reward = interfaceMax.reward
            interfaceMax.send_agent_control(reward_in = reward)

        # Collect tracking data
        tracker.fill_trajectory(state, reward)
        interfaceMax.send_state_to_slider(state, reward)

        # Prepare next cycle
        state = next_state
        action = next_action
        t_idx += 1

        ########################################################
        ##############      TRAIN MODEL      ###################
        ########################################################
        # Train on feedback + exploration_bonus
        if interfaceMax.received and len(agent.delay_memory.buffer) >= agent.reward_length:
            interfaceMax.reward = 0
            interfaceMax.received = False
            rewards = env.set_reward(reward)
            print(str(state) + '; eps = ' + str(agent.eps_threshold))

            agent.remember_rewards(rewards)
            batch = np.reshape(agent.reward_memory.buffer, [agent.reward_length, 3])
            agent.train(sess, batch)

            reward = 0
            rewards *= 0

        # Train on experience (replay memory contains only feedback WITHOUT bonus)
        elif len(agent.replay_memory.buffer) > (2*agent.batch_size):
            batch = agent.replay_memory.sample_random(agent.batch_size)
            agent.train(sess, batch)

        # Train on exploration_bonus
        elif len(agent.delay_memory.buffer) >= agent.reward_length:
            batch = agent.delay_memory.sample(agent.reward_length)
            batch = np.reshape(batch, [agent.reward_length, 3])
            agent.train(sess, batch)

        ########################################################
        ##############          PAUSED       ###################
        ########################################################
        if interfaceMax.paused:
            interfaceMax.send_workflow_control(paused = 1)

            while interfaceMax.paused:
                time.sleep(0.01)

                if interfaceMax.previous:
                    interfaceMax.previous = False
                    if not len(tracker.trajectory) == (abs(interfaceMax.idx)-1):
                        interfaceMax.idx += 1
                        state = tracker.trajectory[-interfaceMax.idx][1].T
                        action, _ = agent.act(sess, state, t_idx)

                        interfaceMax.send_agent_control(previous_s = 1)
                        interfaceMax.send_state(state[0])

                if interfaceMax.next:
                    interfaceMax.next = False
                    if not interfaceMax.idx == 1:
                        interfaceMax.idx -= 1
                        state = tracker.trajectory[-interfaceMax.idx][1].T
                        action, _ = agent.act(sess, state, t_idx)

                        interfaceMax.send_agent_control(next_s = 1)
                        interfaceMax.send_state(state[0])

                if interfaceMax.VSTsample_bool:
                    interfaceMax.VSTsample_bool = False
                    state = interfaceMax.VSTstate
                    state = np.reshape(state,[1,agent.state_size])
                    interfaceMax.send_state(state[0])
                    action, _ = agent.act(sess, state, t_idx)

                # Provide one-state reward
                if interfaceMax.received:
                    reward = copy.deepcopy(interfaceMax.reward)
                    interfaceMax.reward = 0
                    interfaceMax.received = False

                    agent.remember_single_reward(tracker, state, action, reward)
                    agent.train(sess, np.reshape(np.array([state,action,reward]), [1, 3]))

                    next_state = env.step(state, action)
                    next_action, rand_bool = agent.act(sess, next_state, t_idx)
                    agent.remember_transition(state, action)

                    interfaceMax.send_state(next_state[0])
                    interfaceMax.send_workflow_control(rand=rand_bool)
                    interfaceMax.send_agent_control(reward_in=reward)
                    interfaceMax.send_state_to_slider(state, reward)

                    print(str(reward) + ' for action ' + str(action) + ' and state ' + str(state))

                    state = next_state
                    action = next_action
                    reward = 0
                    t_idx += 1

                # Control 1: Explore state
                if interfaceMax.resetstate:
                    interfaceMax.resetstate = False
                    interfaceMax.send_agent_control(explore_state = 1)
                    state, action , t_idx = explore_state(sess, agent, env, tracker, t_idx, interfaceMax)

                # Control 4: Super (dis)like
                if interfaceMax.super_like:
                    interfaceMax.super_like = False
                    interfaceMax.send_agent_control(superlike=interfaceMax.superlike_value)
                    state = interfaceMax.VSTstate
                    state = np.reshape(state, [1, agent.state_size])
                    super_like(agent, env, tracker, state, interfaceMax.superlike_value)

                # Control 5: Explore random action
                if interfaceMax.rnd_action:
                    interfaceMax.rnd_action = False
                    interfaceMax.send_agent_control(explore_action=1)
                    action = explore_random_action(agent, state, t_idx)
                    state = env.step(state, action)
                    interfaceMax.send_state(state[0])

                # Control 6: reset model
                if interfaceMax.resetmodel:
                    interfaceMax.initialise_client(STATE_SIZE, ACTION_SIZE, TRANSITION_TIME, 1)
                    if not TRAINING_LABEL == 'TEST':
                        agent.save_model(sess, save_path, 'model_reset', t_idx)
                        with open('./datalogs/tracker_nb' + str(nb_iter) + '_it' + str(t_idx) + '_reset_' + TRAINING_LABEL + '.pkl', 'wb') as output:
                            pickle.dump(tracker, output, pickle.HIGHEST_PROTOCOL)
                            nb_iter += 1
                    sess, agent, env, tracker = init_program(started_bool = True)
                    interfaceMax.resetmodel = False

                if interfaceMax.save:
                    agent.save_model(sess, save_path, interfaceMax.save_modelname, t_idx)
                    interfaceMax.save = False

                if interfaceMax.load:
                    agent.load_model(sess, interfaceMax.load_modelname)
                    interfaceMax.load = False

                if not interfaceMax.running:
                    break

            interfaceMax.resetstate = False
            interfaceMax.resample_states = False
            interfaceMax.new_speed = False
            interfaceMax.super_like = False
            interfaceMax.rnd_action = False
            interfaceMax.idx = 1
            interfaceMax.send_workflow_control(paused = 0)

        ########################################################
        ##############    AGENT CONTROLS     ###################
        ########################################################
        # Control 1: Explore state
        if interfaceMax.resetstate:
            interfaceMax.resetstate = False
            interfaceMax.send_agent_control(explore_state = 1)
            state, action, t_idx = explore_state(sess, agent, env, tracker, t_idx, interfaceMax)

        # Control 2: Adjust precision (Rescale actions)
        if interfaceMax.resample_states:
            interfaceMax.resample_states = False
            interfaceMax.send_agent_control(precision=1 / env.state_steps)
            resample_actions(env, t_idx, interfaceMax.resample_factor)

        # Control 3: Adjust speed
        # - Adjust reward length
        # - Set new transition time
        if interfaceMax.new_speed:
            interfaceMax.new_speed = False
            interfaceMax.send_agent_control(time = (TRANSITION_TIME*1000))

            adjust_reward_length(agent, t_idx, interfaceMax.increment_reward_length)
            rescale_transitions(agent, t_idx)

        # Control 4: Super (dis)like
        if interfaceMax.super_like:
            interfaceMax.super_like = False
            interfaceMax.send_agent_control(superlike = interfaceMax.superlike_value)

            interfaceMax.send_state(agent.delay_memory.sample(1)[0][0][0])
            super_like(agent, env, tracker, state, interfaceMax.superlike_value)

        # Control 5: Explore action
        if interfaceMax.rnd_action:
            interfaceMax.rnd_action = False
            interfaceMax.send_agent_control(explore_action = 1)
            action = explore_action(agent, state, t_idx)

        # Control 6: reset model
        if interfaceMax.resetmodel:
            interfaceMax.initialise_client(STATE_SIZE, ACTION_SIZE, TRANSITION_TIME, 1)
            if not TRAINING_LABEL == 'TEST':
                agent.save_model(sess, save_path, 'model_reset', t_idx)
                with open('./datalogs/tracker_nb' + str(nb_iter) + '_it' + str(t_idx) + '_reset_' + TRAINING_LABEL + '.pkl','wb') as output:
                    pickle.dump(tracker, output, pickle.HIGHEST_PROTOCOL)
                    nb_iter += 1
            sess, agent, env, tracker = init_program(started_bool = True)
            interfaceMax.resetmodel = False
            interfaceMax.paused = True

    # Save model and end program
    interfaceMax.initialise_client(STATE_SIZE, ACTION_SIZE , TRANSITION_TIME, 1)
    if not TRAINING_LABEL == 'TEST':
        agent.save_model(sess, save_path, 'model_end', t_idx)
        with open('./datalogs/tracker_nb' + str(nb_iter) + '_it' + str(t_idx) + '_end_' + TRAINING_LABEL + '.pkl','wb') as output:
            pickle.dump(tracker, output, pickle.HIGHEST_PROTOCOL)
    print('Data saved at ' + str(os.getcwd()) + '/datalogs/')

    interfaceMax.end_thread()
    sys.exit()



