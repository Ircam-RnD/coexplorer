from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server
import numpy as np
import threading

class OSCClass:
    def __init__(self, STATE_SIZE, ACTION_SIZE , TRANSITION_TIME, ip, port, TRAINING_LABEL):

        self.received = False
        self.resample_states = False
        self.resetstate = False
        self.new_speed = False
        self.super_like = False
        self.rnd_action = False
        self.previous = False
        self.next = False
        self.VSTsample_bool = False
        self.get_trajlist = False

        self.running = True
        self.paused = True
        self.save = False
        self.load = False
        self.resetmodel = False

        self.reward = 0
        self.resample_factor = 0
        self.superlike_value = 0
        self.increment_reward_length = 0
        self.state_idx = 0
        self.row1_idx = 0
        self.col1_idx = 0
        self.row2_idx = 0
        self.col2_sl_idx = 0
        self.col2_sdl_idx = 0
        self.col2_es_idx = 0
        self.idx = 1
        self.VSTstate = 0
        self.save_modelname = None


        self.client = udp_client.SimpleUDPClient('localhost', 8000)
        self.initialise_client(STATE_SIZE, ACTION_SIZE , TRANSITION_TIME, 1)

        self.dispatch = dispatcher.Dispatcher()

        # Main controls
        self.dispatch.map("/reward", self.store_reward)

        # Workflow controls
        self.dispatch.map("/stop_program", self.stop_program)
        self.dispatch.map("/pause", self.pause_training)
        self.dispatch.map("/previous_state", self.previous_state)
        self.dispatch.map("/next_state", self.next_state)
        self.dispatch.map("/sample_vst", self.sample_vststate)
        self.dispatch.map("/save_model", self.save_model)
        self.dispatch.map("/load_model", self.load_model)
        self.dispatch.map("/reset_model", self.reset_model)

        # Agent controls
        self.dispatch.map("/resample", self.adjust_sampling)
        self.dispatch.map("/super_like", self.record_superlike)
        self.dispatch.map("/explore_state", self.reset_state)
        self.dispatch.map("/explore_action", self.random_action)
        self.dispatch.map("/new_speed", self.rescale_reward_length)

        self.server = osc_server.ThreadingOSCUDPServer((ip, port), self.dispatch)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.start()

    # ------------------        CLIENT          ------------------------------
    def initialise_client(self, STATE_SIZE, ACTION_SIZE , TRANSITION_TIME, paused_bool):
        self.send_state(np.ones(STATE_SIZE)*0.5)
        self.send_agent_control(time = TRANSITION_TIME * 1000)
        self.send_agent_control(precision = 1.0/ACTION_SIZE)
        self.send_workflow_control(training = 0)
        self.send_workflow_control(paused = paused_bool)
        self.state_idx = 0
        self.row1_idx = 0
        self.col1_idx = 0
        self.row2_idx = 0
        self.col2_sl_idx = 0
        self.col2_sdl_idx = 0
        self.col2_es_idx = 0

    def send_state(self, state):
        self.client.send_message('/params', state)

    def send_state_to_slider(self, state, label):
        state = str(state[0])[1:-1]

        self.col1_idx = self.state_idx % 278
        content_slider1 = 'set ' + str(self.col1_idx) + ' ' + str(self.row1_idx) + ' ' + str(state)

        if label == 'Superlike':
            self.row2_idx = 0
            format_slider1 = 'cell ' + str(self.col1_idx) + ' ' + str(self.row1_idx) + ' brgb 16 127 64'
            content_slider2 = 'set ' + str(self.col2_sl_idx) + ' 0 ' + state
            format_slider2 = 'cell ' + str(self.col2_sl_idx) + ' 0 brgb 16 127 64'
            self.col2_sl_idx += 1

        elif label == 'Superdislike':
            self.row2_idx = 1
            format_slider1 = 'cell ' + str(self.col1_idx) + ' ' + str(self.row1_idx) + ' brgb 127 0 2'
            content_slider2 = 'set ' + str(self.col2_sdl_idx) + ' 1 ' + state
            format_slider2 = 'cell ' + str(self.col2_sdl_idx) + ' 1 brgb 127 0 2'
            self.col2_sdl_idx += 1

        elif label == 'Explore_state':
            self.row2_idx = 2
            format_slider1 = 'cell ' + str(self.col1_idx) + ' ' + str(self.row1_idx) + ' brgb 253 204 101'
            content_slider2 = 'set ' + str(self.col2_es_idx) + ' 2 ' + state
            format_slider2 = 'cell ' + str(self.col2_es_idx) + ' 2 brgb 253 204 101'
            self.col2_es_idx += 1

        elif label == 1:
            format_slider1 = 'cell ' + str(self.col1_idx) + ' ' + str(self.row1_idx) + ' brgb 150 255 151'
            content_slider2 = ''
            format_slider2 = ''

        elif label == -1:
            format_slider1 = 'cell ' + str(self.col1_idx) + ' ' + str(self.row1_idx) + ' brgb 252 121 132'
            content_slider2 = ''
            format_slider2 = ''

        else:
            format_slider1 = 'cell ' + str(self.col1_idx) + ' ' + str(self.row1_idx) + ' brgb 33 35 34'
            content_slider2 = ''
            format_slider2 = ''

        self.state_idx += 1
        if self.state_idx % 278 == 0:
            self.row1_idx += 1
            self.client.send_message('/toslider1', 'rows ' + str(self.row1_idx+1))
        self.col1_idx += 1

        self.client.send_message('/toslider1', content_slider1)
        self.client.send_message('/toslider1', format_slider1)
        if not content_slider2 == '':
            self.client.send_message('/toslider2', content_slider2)
            self.client.send_message('/toslider2', format_slider2)

    def send_agent_control(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'reward_in':
                self.client.send_message('/reward_in', value)
            elif key == 'time':
                self.client.send_message('/time', value)
            elif key == 'precision':
                self.client.send_message('/precision', value)
            elif key == 'superlike':
                self.client.send_message('/superlike', self.superlike_value)
            elif key == 'explore_state':
                self.client.send_message('/explore_state', value)
            elif key == 'explore_action':
                self.client.send_message('/explore_action', value)
            elif key == 'previous_s':
                self.client.send_message('/previous_s', value)
            elif key == 'next_s':
                self.client.send_message('/next_s', value)

    def send_workflow_control(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'init':
                self.client.send_message('/init', value)
            elif key == 'paused':
                value = (value + 1) % 2
                self.client.send_message('/paused', value)
            elif key == 'rand':
                self.client.send_message('/rand', value)

    # ------------------        SERVER          ------------------------------
    # ------------------ Basic reward control   ------------------------------
    def store_reward(self,unused_addr, reward):
        self.reward = reward
        self.received = True

    # ------------------ Workflow controls      ------------------------------
    def pause_training(self, unused_addr, pause_bool):
        self.paused = pause_bool

    def save_model(self, unused_addr, *args):
        self.save_modelname = args[1]
        self.save = True

    def load_model(self, unused_addr, model_name):
        self.load_modelname = model_name
        self.load = True

    def reset_model(self, unused_addr, reset_bool):
        self.resetmodel = True

    def stop_program(self, unused_addr, running_bool):
        self.running = running_bool

    def end_thread(self):
        self.server.shutdown()

    # ------------------ Agent controls         ------------------------------
    def adjust_sampling(self,unused_addr, resample):
        self.resample_factor = pow(2,resample)
        self.resample_states = True

    def record_superlike(self,unused_addr, superlike_flag):
        self.superlike_value = superlike_flag
        self.super_like = True

    def reset_state(self,unused_addr, reset):
        self.resetstate = True

    def random_action(self,unused_addr, action):
        self.rnd_action = True

    def rescale_reward_length(self, unused_addr, new_reward_length):
        self.increment_reward_length = new_reward_length * -2
        self.new_speed = True

    # ------------------ Workflow controls (paused) --------------------------
    def previous_state(self, unused_addr, previous_bool):
        self.previous = True

    def next_state(self, unused_addr, next_bool):
        self.next = True

    def sample_vststate(self, unused_addr, *args):
        VSTsample = np.array(args)
        self.VSTstate = VSTsample
        self.VSTsample_bool = True




