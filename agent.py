import os
import copy
import numpy as np
from Tiles.tiles import *
import tensorflow as tf

class Memory(object):
    def __init__(self, buffer_size, state_size):
        self.buffer = []
        self.state_size = state_size
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(self.buffer[:size]), [size, 3])

    def sample_random(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 3])

class DTAMERAgent:
    def __init__(self, STATE_SIZE, ACTION_SIZE, HIDDEN_LAYER_NB, HIDDEN_LAYER_SIZE, EPS_DECAY, LEARNING_RATE, REWARD_LENGTH, REWARD, TRANSITION_TIME, REPLAY_SIZE, BATCH_SIZE, EPS_START):

        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.reward_length = REWARD_LENGTH
        self.reward_size = REWARD
        self.transition_time = TRANSITION_TIME
        self.epsilon_decay = EPS_DECAY
        self.learning_rate = tf.Variable(LEARNING_RATE, trainable=False)
        self.model = 'DTAMER'
        self.state_ranges = [(0.0,1.0)]* self.state_size

        self.replay_size = REPLAY_SIZE
        self.batch_size = BATCH_SIZE
        self.epsilon_start = EPS_START
        self.epsilon_end = 0.0
        self.exploration_constant = 0.01
        self.numtilings = 64 # Based on Sutton p.178 (nb = pow2 > 4k with k = sdim)
        self.tile_size = 0.4 # With precision = tilesize/numtilings = 0.3/64 =  0.046875

        # To assure non-collision, hashtable_size = pow(1/tilesize,sdim)*numtilings
        self.hashtable_size = int(min(200000, pow(1 / self.tile_size, self.state_size) * self.numtilings))

        self.eps_threshold = EPS_START
        self.average_reward = 0
        self.time_idx = 0
        self.density_weights = np.ones(self.hashtable_size)*(1.0/self.hashtable_size)
        self.reward_memory = Memory(self.reward_length, 3)
        self.replay_memory = Memory(self.replay_size, 3)
        self.delay_memory = Memory(int(np.floor(0.2 / self.transition_time + self.reward_length)), 3)
        self._build_qnetwork(HIDDEN_LAYER_NB, HIDDEN_LAYER_SIZE)
        self.saver = tf.train.Saver()

    def fc_layer(self, input, size_in, size_out, name="fc"):
        with tf.name_scope(name):

            #initializer = tf.contrib.layers.xavier_initializer()
            initializer = tf.initializers.truncated_normal(stddev=0.3)
            #initializer = tf.initializers.random_normal(stddev=0.3)
            # initializer = tf.initializers.random_uniform(minval=-0.4, maxval=0.4)
            w = tf.Variable(initializer([size_in, size_out]), name="W")
            b = tf.Variable(tf.constant(0., shape=[size_out]), name="B")
            act = tf.matmul(input, w) + b
            #tf.summary.histogram("weights", w)
            #tf.summary.histogram("biases", b)
            #tf.summary.histogram("activations", act)
            return act

    def _build_qnetwork(self, HIDDEN_LAYER_NB, HIDDEN_LAYER_SIZE):

        self.scalarInput = tf.subtract(tf.placeholder(shape=[None, self.state_size], dtype=tf.float32), 0.5)

        if HIDDEN_LAYER_NB == 1:
            fc1 = self.fc_layer(self.scalarInput, self.state_size, HIDDEN_LAYER_SIZE, "fc1")
            relu1 = tf.nn.relu(fc1)
            #tf.summary.histogram("fc1/relu1", relu1)
            self.act_values = self.fc_layer(relu1, HIDDEN_LAYER_SIZE, self.action_size, "fc2")

        elif HIDDEN_LAYER_NB == 2 :
            fc1 = self.fc_layer(self.scalarInput, self.state_size, HIDDEN_LAYER_SIZE, "fc1")
            relu1 = tf.nn.relu(fc1)
            #tf.summary.histogram("fc1/relu1", relu1)
            fc2 = self.fc_layer(relu1, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, "fc2")
            relu2 = tf.nn.relu(fc2)
            self.act_values = self.fc_layer(relu2, HIDDEN_LAYER_SIZE, self.action_size, "fc3")
            #tf.summary.histogram("fc2/relu2", relu2)

        elif HIDDEN_LAYER_NB == 3:
            fc1 = self.fc_layer(self.scalarInput, self.state_size, HIDDEN_LAYER_SIZE, "fc1")
            relu1 = tf.nn.relu(fc1)
            #tf.summary.histogram("fc1/relu1", relu1)
            fc2 = self.fc_layer(relu1, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, "fc2")
            relu2 = tf.nn.relu(fc2)
            #tf.summary.histogram("fc2/relu2", relu2)
            fc3 = self.fc_layer(relu2, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, "fc3")
            relu3 = tf.nn.relu(fc3)
            self.act_values = self.fc_layer(relu3, HIDDEN_LAYER_SIZE, self.action_size, "fc4")
            #tf.summary.histogram("fc3/relu3", relu3)
        else:
            exit()

        with tf.name_scope("loss"):
            self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.weights = tf.placeholder(shape=[None], dtype=tf.float32)
            actions_onehot = tf.one_hot(self.actions, self.action_size, dtype=tf.float32)

            self.Q = tf.reduce_sum(tf.multiply(self.act_values, actions_onehot), axis=1)
            self.td_error = self.weights * tf.square(self.targetQ - self.Q)
            self.loss = tf.reduce_mean(self.td_error)
            #tf.summary.scalar("loss", self.loss)

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            self.train_step = optimizer.apply_gradients(grads)
            #for index, grad in enumerate(grads):
                #tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])

        #self.summ = tf.summary.merge_all()

    def remember_transition(self, state, action):
        tiles_idx = self.calc_tiles_idx(state[0])
        state_prob = np.sum(self.density_weights[tiles_idx]) / ((self.time_idx + 1) * self.numtilings + 1)
        self.density_weights[tiles_idx] += 1.0
        state_next_prob = np.sum(self.density_weights[tiles_idx]) / ((self.time_idx + 2) * self.numtilings + 1)
        pseudo_totalcount = (1 - state_next_prob) / (state_next_prob - state_prob) * state_prob
        pseudo_count = pseudo_totalcount * state_prob
        reward = np.clip(self.eps_threshold * pow((pseudo_count + self.exploration_constant), -0.5),0,2)

        self.delay_memory.add(np.reshape(np.array([state, action, reward]), [1, 3]))

        tiles_idx = self.calc_tiles_idx(state[0])
        self.density_weights[tiles_idx] += 1.0

    def calc_tiles_idx(self, state):
        return tiles(self.numtilings, self.hashtable_size, state / self.tile_size)

    def remember_rewards(self, rewards):
        self.reward_memory.buffer = self.delay_memory.sample(self.reward_length)

        temp = copy.deepcopy(self.reward_memory.buffer)
        temp[:, 2] = rewards
        self.replay_memory.add(temp)

        self.reward_memory.buffer[:, 2] += rewards
        self.reward_memory.buffer = list(self.reward_memory.buffer)

    def remember_single_reward(self, tracker, state, action, reward):
        temp = np.reshape(np.array([state, action, reward]), [1, 3])
        self.replay_memory.add(temp)
        tracker.fill_trajectory(temp[0,0], temp[0,2])

    def act(self, sess, state, t = 0):

        self.time_idx = t

        if self.epsilon_decay == 0:
            self.eps_threshold = self.epsilon_end
        else:
            self.eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * t / self.epsilon_decay)

        invalid_actions = [ind*2+1 if x == 0 else ind*2 for ind,x in enumerate(state[0]) if x in (0,1)]
        #invalid_actions = [ind*2+1 if x <= range_state[ind][0] else ind*2 for ind,x in enumerate(state[0]) if x <= self.state_ranges[ind][0] or x >= self.state_ranges[ind][1]]
        valid_actions = [x for x in np.arange(self.state_size*2) if x not in invalid_actions]

        if np.random.rand() <= self.eps_threshold:
            action = valid_actions[random.randrange(len(valid_actions))]
            rand_bool = True
            return action, rand_bool
        else:
            act_values = sess.run(self.act_values, feed_dict={self.scalarInput: state})[0]
            act_values[invalid_actions] = np.min(act_values)
            action = np.argmax(act_values)
            rand_bool = False
            return action, rand_bool

    def train(self, sess, batch):

        new_estimate = batch[:, 2]
        # old_estimate = sess.run(self.act_values, feed_dict={self.scalarInput: np.vstack(batch[:, 0])})[np.arange(len(batch)),batch[:,1].astype(int)]
        # new_estimate = batch[:, 2] - self.average_reward
        # self.average_reward += self.beta * np.sum(new_estimate - old_estimate)

        sess.run(self.train_step, feed_dict={self.scalarInput: np.vstack(batch[:, 0]), self.targetQ: new_estimate, self.actions: batch[:, 1], self.weights: np.ones(batch.shape[0])})

    def load_model(self, sess, label):
        filename = label.split(':')[1].split('.data')[0]
        self.saver.restore(sess, filename)
        print('Load successful!')

    def save_model(self, sess, save_path, label, t_idx):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.saver.save(sess, save_path + '/' + label + '.ckpt', t_idx)
