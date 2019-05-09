import time
import numpy as np

class Tracker:

    def __init__(self,state_size, state_steps, TRAINING_LABEL):

        # RL
        self.trajectory = []

        self.state_steps = state_steps
        self.state_size = state_size
        self.label = TRAINING_LABEL

    def fill_trajectory(self, state, label):
        self.trajectory.append((time.time(), state.T, label))

    def save_trajectory(self):

        idx_interacted_states = np.reshape(np.arange(len(self.interacted_states)),[len(self.interacted_states),1])
        temp_interacted_states = np.hstack((np.reshape(np.array(self.interacted_states), [len(self.interacted_states), 2]),idx_interacted_states))
        temp_interacted_states_sorted = np.flip(temp_interacted_states[temp_interacted_states[:, 1].astype(np.str).argsort()],axis=0)

        with open('./trajectory.txt', 'w') as f:
            f.write('0, set 0 0 State ;\n')
            f.write('1, set 0 1 Label ;\n')
            idx = 1
            line_nb = 2
            for line in temp_interacted_states:
                f.write(str(line_nb) + ', ' + 'set ' + str(idx) + ' 0 ' +  str(np.array_str(line[0][0])[1:-1]) + ' ;' + '\n')
                if line[1] == 'Superlike':
                    f.write(str(line_nb + 1) + ', ' + 'cell ' + str(idx) + ' 0 brgb 16 127 64'  + ' ;' + '\n')
                elif line[1] == 'Superdislike':
                    f.write(str(line_nb + 1) + ', ' + 'cell ' + str(idx) + ' 0 brgb 127 0 2'  + ' ;' + '\n')
                elif line[1] == 'Explore_state':
                    f.write(str(line_nb + 1) + ', ' + 'cell ' + str(idx) + ' 0 brgb 253 204 101'  + ' ;' + '\n')
                else:
                    f.write(str(line_nb + 1) + ', ' + 'cell ' + str(idx) + ' 0 brgb 33 35 34' + ' ;' + '\n')
                idx += 1
                line_nb += 2

        with open('./trajectory_sorted.txt', 'w') as f:
            f.write('0, set 0 0 State ;\n')
            f.write('1, set 0 1 Label ;\n')
            idx = 1
            line_nb = 2
            for line in temp_interacted_states_sorted:
                f.write(str(line_nb) + ', ' + 'set ' + str(idx) + ' 0 ' + str(np.array_str(line[0][0])[1:-1]) + ' ;' + '\n')
                if line[1] == 'Superlike':
                    f.write(str(line_nb + 1) + ', ' + 'cell ' + str(idx) + ' 0 brgb 16 127 64'  + ' ;' + '\n')
                elif line[1] == 'Superdislike':
                    f.write(str(line_nb + 1) + ', ' + 'cell ' + str(idx) + ' 0 brgb 127 0 2'  + ' ;' + '\n')
                elif line[1] == 'Explore_state':
                    f.write(str(line_nb + 1) + ', ' + 'cell ' + str(idx) + ' 0 brgb 253 204 101'  + ' ;' + '\n')
                else:
                    f.write(str(line_nb + 1) + ', ' + 'cell ' + str(idx) + ' 0 brgb 33 35 34' + ' ;' + '\n')
                idx += 1
                line_nb += 2

