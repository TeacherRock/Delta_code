import numpy as np
import math
import random
from utils.motor_matlab import PPI_matlab
import matlab.engine
from collections import deque
from itertools import islice
import time
import pandas as pd
import os



PID_control_matlab = PPI_matlab()
# PID_control_matlab.sampT = 1 / 16000
# activate matlab engine
matlab_engine = matlab.engine.start_matlab()
matlab_engine.eval("warning('off', 'all');", nargout=0)
print("matlab engine activate done")


class Environment:
    def __init__(self, state_dim, action_dim, results_folder=''):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self._action = np.zeros([1, action_dim]).reshape(-1)

        # Kvp：150 - 350

        self.random_Kvp = np.random.uniform(100.0, 1500.0)
        self.random_Kpp = np.random.uniform(100.0, self.random_Kvp)
        self.random_Kvi = 0.0

        self.controller_param_reset = np.array([1000.0, 1500.0, 30.0])
        self.controller_param_init = self.controller_param_reset
        action_percnetage = 0.05
        self.action_upper_bound =  self.controller_param_init * action_percnetage
        self.action_lower_bound = -self.controller_param_init * action_percnetage
        self.K_strict_upper_bound = np.array([1000.0, 1500.0, 30.0])
        self.K_strict_lower_bound = np.array([20.0, 20.0, 0.0])
        self.K_upper_bound = np.array([950.0, 950.0, 10.0])
        self.K_lower_bound = np.array([150.0, 150.0, 0.0])

        self.controller_param = self.controller_param_init

        self.boundary_position = [-40.0, 40.0]

        self.training = False

        self.BW_Current = 2000.0

        self.sys_num = 0
        # Sys 1
        self.J1 = 45e-6
        self.J2 = 71.959e-6
        self.K12 = 1830.016
        self.C12 = 0.03844
        self.J  = self.J1 + self.J2 # 控制器參數 (for 正規化 Kvp)
        self.fc = [-0.033249, 0.027123]
        self.fs = [-0.04, 0.04]
        self.B_sys = 0.0003457395

        self.V_max = 2000 * 2*math.pi / 60.0  # 4200 (rpm)
        self.torque_disturb = self.B_sys*self.V_max
        self.tau_max = 2.54 - self.torque_disturb # 2.54 (Nm)
        self.A_max = self.tau_max / self.J

        # self.change_env = "static"
        # self.change_env = "random"
        self.change_env = "random_J"

        self.first_step = True
        self.results_folder = results_folder

        self.reward_queue = CustomQueue(10)  # For check the terminal condition

        self.matlab_clear = 1
        self.check_K_bound = False

        self.reward_info = []
        self.reward_list_record = []

        self.terminal_condition_type = ""

        self.random_MO = 5.0
        self.random_T_load = 0.0
        current_dir = os.path.dirname(__file__)
        csv_path = os.path.join(current_dir, "..", "..", "..", "system_parameters", "system_parameters_table.csv")
        csv_path = os.path.abspath(csv_path)
        self.sys_settings = pd.read_csv(csv_path)
        self.sys_settings = self.sys_settings.dropna()
        # self.sys_settings = self.sys_settings[(self.sys_settings['sys_num']>=21) & (self.sys_settings['sys_num']<=30)]

    def reset(self, ini_parameter=None):
        if self.matlab_clear % 300 == 0:
            global matlab_engine
            matlab_engine.quit()
            print("\nsleep")
            time.sleep(20)
            matlab_engine = matlab.engine.start_matlab()
            matlab_engine.eval("warning('off', 'all');", nargout=0)

        self.matlab_clear = self.matlab_clear + 1

        state = np.zeros(self.state_dim)
        for i in range(self.state_dim):
            state[i] = 0

        # random initial parameters
        if ini_parameter is None:
            # self.random_Kvp = np.random.uniform(self.K_lower_bound[1], self.K_upper_bound[1])
            # self.random_Kpp = np.random.uniform(self.K_lower_bound[0], self.random_Kvp)
            self.controller_param_init = np.array([self.random_Kpp, self.random_Kvp, self.random_Kvi])
            self.random_MO  = 5.0
        else:
            self.controller_param_init = np.array(ini_parameter).reshape(-1)
        self.controller_param = self.controller_param_init

        reward_info = self.env_sim(self.controller_param)

        self.thresholds_setting = {
            "settling_time"      : reward_info["settling_time"],   
            "Emax_before"        : reward_info["Emax_before"], 
            "Emax_after"         : reward_info["Emax_after"],   
            "Eavg_before"        : reward_info["Eavg_before"], 
            "Eavg_after"         : reward_info["Eavg_after"], 
            "max_torque_flag"    : reward_info["max_torque_flag"], 
            "GM_velocity"        : reward_info["GM_velocity"],
            "settling_time_001p" : reward_info["settling_time_001p"],
            "settling_time_01p"  : reward_info["settling_time_01p"],
            "settling_time_1p"   : reward_info["settling_time_1p"],
            "settling_time_3p"   : reward_info["settling_time_3p"],
            "damping_ratio"      : reward_info["damping_ratio"],
            "Ess_step"           : reward_info["Ess_step"], 
            "Ess_trap"           : reward_info["Ess_trap"],
            "overshoot_load"     : reward_info["overshoot_load"], 
            "settling_time_load" : reward_info["settling_time_load"]
        }
        self.first_GM_velocity = reward_info["GM_velocity"]

        # self.GM_velocity_threshold = np.random.uniform(5.0, 10.0)
        self.GM_velocity_threshold = 10.0

        reward = self.cal_reward(reward_info)
        self.reward_info = reward_info
        next_state = self.set_state()
        self.reward_queue.empty()
        self.terminal_flag = False
        self.check_K_bound = False
        
        self.reward_list_record = []
        self.first_step =False
        self.terminal_condition_type = ""
        
        self._action = np.zeros([1, self.action_dim]).reshape(-1)
        
        return next_state

    def step(self, state, action):
        if self.training:
            action = self.apply_noise(action)
        self.normalize_action(action)

        ## if all action is small, stop
        if all(np.abs(self._action) < 0.01):
            self.terminal_condition_type = "small action"
            self.terminal_flag = True

        self.controller_param += self._action
        reward_info = self.env_sim(self.controller_param)
        reward = self.cal_reward(reward_info)
        next_state = self.set_state()

        ## Rewar thrsshold comapre to the last one

        self.thresholds_setting = {
            "settling_time"      : reward_info["settling_time"],   
            "Emax_before"        : reward_info["Emax_before"], 
            "Emax_after"         : reward_info["Emax_after"],   
            "Eavg_before"        : reward_info["Eavg_before"], 
            "Eavg_after"         : reward_info["Eavg_after"], 
            "max_torque_flag"    : reward_info["max_torque_flag"], 
            "GM_velocity"        : reward_info["GM_velocity"],
            "settling_time_001p" : reward_info["settling_time_001p"],
            "settling_time_01p"  : reward_info["settling_time_01p"],
            "settling_time_1p"   : reward_info["settling_time_1p"],
            "settling_time_3p"   : reward_info["settling_time_3p"],
            "damping_ratio"      : reward_info["damping_ratio"],
            "Ess_step"           : reward_info["Ess_step"], 
            "Ess_trap"           : reward_info["Ess_trap"],
            "overshoot_load"     : reward_info["overshoot_load"], 
            "settling_time_load" : reward_info["settling_time_load"]
        }

        self.first_GM_velocity = reward_info["GM_velocity"]
        self.reward_info = reward_info

        return  state, next_state, reward

    def env_sim(self, controller_param):

        # Feedback Controller parameter
        PID_control_matlab.Kpp = controller_param[0]
        PID_control_matlab.Kvp = controller_param[1]
        PID_control_matlab.Kvi = controller_param[2]

        matlab_engine.workspace["BW_Current"] = self.BW_Current
        matlab_engine.workspace["J1"]    = self.J1
        matlab_engine.workspace["J2"]    = self.J2
        matlab_engine.workspace["C12"]   = self.C12
        matlab_engine.workspace["K12"]   = self.K12
        matlab_engine.workspace["J"]     = self.J
        matlab_engine.workspace["B_sys"] = self.B_sys
        matlab_engine.workspace["Vmax"]  = self.V_max
        matlab_engine.workspace["Amax"]  = self.A_max
        matlab_engine.workspace["fc"] = matlab.double(self.fc)
        matlab_engine.workspace["fs"] = matlab.double(self.fs)
        matlab_engine.workspace["v_fric"]  = 0.0
        matlab_engine.workspace["T_load"] = self.random_T_load

        error,     _, _, _, _ = PID_control_matlab.controller_step(self.start_position, 1.0, matlab_engine)

        matlab_engine.workspace["BW_Current"] = self.BW_Current
        matlab_engine.workspace["J1"]    = self.J1
        matlab_engine.workspace["J2"]    = self.J2
        matlab_engine.workspace["C12"]   = self.C12
        matlab_engine.workspace["K12"]   = self.K12
        matlab_engine.workspace["J"]     = self.J
        matlab_engine.workspace["B_sys"] = self.B_sys
        matlab_engine.workspace["Vmax"]  = self.V_max
        matlab_engine.workspace["Amax"]  = self.A_max
        matlab_engine.workspace["fc"] = matlab.double(self.fc)
        matlab_engine.workspace["fs"] = matlab.double(self.fs)
        matlab_engine.workspace["v_fric"]  = 0.0
        matlab_engine.workspace["T_load"] = self.random_T_load

        error_acc, _, _, _, _ = PID_control_matlab.controller_trap(self.start_position, self.goal_position, matlab_engine)

        _reward_info = {}
        _reward_info.update(error)
        _reward_info.update(error_acc)
        actural_reward_list = []

        reward_names = [
            "overshoot", "overshoot_load", "settling_time", "settling_time_load", \
            "max_torque_flag", "GM_velocity",\
            "Emax_before", "Eavg_before", "Emax_after", "Eavg_after",\
            "settling_time_001p", "settling_time_01p", "settling_time_1p", "settling_time_3p", \
            "damping_ratio", "Ess_step", "Ess_trap"
        ]
        
        temp_actual_reward = []
        for name in reward_names:
            temp_actual_reward.append(_reward_info[name])
        self.reward_list_record.append(temp_actual_reward)
        return _reward_info
    
    def set_state(self):
        state = np.zeros(self.state_dim)
        
        # Kpp, Kvp, Kvi
        state[0] = self.controller_param[0] / 100.0
        state[1] = self.controller_param[1] / 100.0
        state[2] = self.controller_param[2] / 10.0
        
        # DKpp, DKvp, DKvi
        state[3] = self._action[0] / 10.0 # Normalize to [0, 5] from [0, 50]
        state[4] = self._action[1] / 10.0 # Normalize to [0, 5] from [0, 50]
        state[5] = self._action[2] / 10.0 # Normalize to [0, 5] from [0, 50]

        keys = ["overshoot", "settling_time", "GM_velocity"]
        state[6] = self.reward_info[keys[0]] / 10.0  # Normalize to [0, 1] from [0, 0.01]
        state[7] = self.reward_info[keys[1]] * 10.0   # Normalize to [0, 5] from [0, 0.5]
        state[8] = 0.0 \
                    if math.isnan(self.reward_info[keys[2]]) or self.reward_info[keys[2]] == math.inf \
                                            else self.reward_info[keys[2]] / 10.0 # Normalize to [0, 5] from [0, 50]
        return state
    
    def change_system_param(self, sys_num=None):
        if sys_num is None:
            system_choose = self.sys_settings.sample(n=1).iloc[0]
        else:
            system_choose = self.sys_settings[self.sys_settings['sys_num'] == sys_num].iloc[0]

        self.sys_num = system_choose['sys_num'] 
        
        self.J1  = system_choose['J1']
        self.J2  = system_choose['J2']
        self.K12 = system_choose['K12']
        self.C12 = system_choose['C12']
        self.war = system_choose['war']
        self.wr =  system_choose['wr']
        self.J   = self.J1 + self.J2
        self.fc  = [system_choose['fc_min'], system_choose['fc_max']]
        self.fs  = [system_choose['fs_min'], system_choose['fs_max']]
        self.B_sys = system_choose['B_sys']

        self.V_max = 2000 * 2*math.pi / 60.0  # 2000 (rpm)
        self.tau_max = 2.54 - self.B_sys*self.V_max
        self.A_max = self.tau_max / self.J

        self.command_distance = 0.1 * self.V_max + self.V_max**2 / self.A_max # assmue t2 = 0.1s
        self.start_position = 0.0
        self.goal_position  = self.command_distance - self.start_position

        self.random_Kvp = np.random.uniform(system_choose['min_Kvp'], system_choose['max_Kvp'])
        self.random_Kpp = np.random.uniform(system_choose['min_Kpp'], self.random_Kvp)
        self.random_Kvi = 0.0
        self.random_T_load = 0.5

    def cal_reward(self, reward_info):      

        def calculate_cost(value, threshold, max_cost=10.0):
            # return max_cost if value > threshold else value / threshold
            try:
                return_value = 1.0 * (threshold / value - 1)
                return_value =  1.0 if return_value == math.inf or return_value > 1.0 \
                                        else return_value
                return_value = -1.0 if math.isnan(return_value) else return_value
            except:
                if value == 0.0:
                    return_value = 1.0
                else:
                    return_value = -1.0

            if value == 0.0 and threshold == 0.0:
                    return_value = 0.0
                
            return return_value
        
        reward = 0.0
        self.current_reward = []

        cost = -reward_info["overshoot"]/10.0 if reward_info["overshoot"] > self.random_MO else 0.0
        if cost < -5.0:
            cost = -5.0
        reward += cost
        self.current_reward.append(cost)
        # ^ len(self.current_reward) = 1

        thresholds = self.thresholds_setting

        weights= {
            "settling_time"      : 5.0,   
            "Emax_before"        : 0.0, 
            "Emax_after"         : 0.0,   
            "Eavg_before"        : 0.0, 
            "Eavg_after"         : 0.0, 
            "max_torque_flag"    : 10.0, 
            "GM_velocity"        : 0.0,
            "settling_time_001p" : 0.0,
            "settling_time_01p"  : 0.0,
            "settling_time_1p"   : 0.0,
            "settling_time_3p"   : 0.0,
            "damping_ratio"      : 0.0,
            "Ess_step"           : 0.0, 
            "Ess_trap"           : 0.0,
            "overshoot_load"     : 0.0, 
            "settling_time_load" : 5.0
        } # num = 16

        for key in thresholds:
            cost = calculate_cost(reward_info[key], thresholds[key])
            weighted_cost = weights[key] * cost
            reward += weighted_cost
            self.current_reward.append(cost)
        # ^ len(self.current_reward) = 17

        max_torque_cost = -reward_info["max_torque_flag"] * 0.0 \
                                    if reward_info["max_torque_flag"] > 0 else 0.0
        max_torque_cost = -10.0 if max_torque_cost < -10.0 else max_torque_cost
        reward += max_torque_cost
        self.current_reward.append(max_torque_cost)
        # ^ len(self.current_reward) = 18

        if reward_info["GM_velocity"] < self.GM_velocity_threshold and reward_info["GM_velocity"] < self.first_GM_velocity:
            GM_velocity_cost = (reward_info["GM_velocity"] - self.GM_velocity_threshold) * 2.0
        elif reward_info["GM_velocity"] > self.GM_velocity_threshold and reward_info["GM_velocity"] > self.first_GM_velocity:
            GM_velocity_cost = (self.GM_velocity_threshold - reward_info["GM_velocity"]) * 0.1
        else:
            GM_velocity_cost = 0.0
            
        if GM_velocity_cost < -5:
            GM_velocity_cost = -5
        reward = reward + GM_velocity_cost

        if reward_info["GM_velocity"] < 2:
            self.check_K_bound = True
            reward = reward - 5.0 

        # self.current_reward.append(GM_position_cost)
        self.current_reward.append(GM_velocity_cost)
        # ^ len(self.current_reward) = 19

        if np.any(self.controller_param > self.K_strict_upper_bound) \
                or np.any(self.controller_param[:2] < self.K_strict_lower_bound[:2]):
            self.check_K_bound = True
            reward = reward - 1.0

        if self.controller_param[2] < self.K_strict_lower_bound[2]:
            reward = reward - 1.0

        if self.controller_param[1] < self.controller_param[0]:
            reward = reward - 0.5

        cost = -reward_info["Ess_step"]*20.0 if reward_info["Ess_step"] > 0.01 else 0.0
        if cost < -5.0:
            cost = -5.0
        reward += cost
        self.current_reward.append(cost)
        # ^ len(self.current_reward) = 20

        cost = -reward_info["Ess_trap"]*20.0 if reward_info["Ess_trap"] > 0.01 else 0.0
        if cost < -5.0:
            cost = -5.0
        reward += cost
        self.current_reward.append(cost)
        # ^ len(self.current_reward) = 21


        cost = -reward_info["overshoot_load"]/10.0 if reward_info["overshoot_load"] > self.random_MO else 0.0
        if cost < -5.0:
            cost = -5.0
        reward += cost
        self.current_reward.append(cost)
        # ^ len(self.current_reward) = 22

        self.current_reward.append(reward)
        # ^ len(self.current_reward) = 23

        return reward

    def normalize_action(self, action):

        # print("action : ", action)
        # map the action from [-1, 1] to [self.lower_bound, self.upper_bound]
        self._action = (self.action_upper_bound - self.action_lower_bound) * (action[: self.action_dim] + 1) / 2 \
                            + self.action_lower_bound
        
        self._action = np.clip(self._action, self.action_lower_bound, self.action_upper_bound)

        # if any(np.isnan(self._action)):
        #     self._action = np.array([])

    def apply_noise(self, action):
        noise = np.random.normal(0, 0.05, self.action_dim)
        return action[: self.action_dim] + noise

    def goal_select(self, state, bound):
        pos = state[0]
        goal_position = bound[0] if abs(pos - bound[0]) > abs(pos - bound[1]) else bound[1]

        return goal_position
    
    def check_terminal_condition(self):
        if self.reward_queue.counter >= 3.0:
            self.terminal_condition_type = "reward is getting worse"
            return True 
        elif self.check_K_bound == True:
            self.terminal_condition_type = "exceed parameter bound"
            return True
        return self.terminal_flag
    

class CustomQueue:
    def __init__(self, max_size):
        self.queue = deque()
        self.max_size = max_size
        self.counter = 0.0

    def push(self, num):
        if len(self.queue) >= self.max_size:
            self.queue.popleft()
        
        self.queue.append(num)

    def get_queue(self):
        return list(self.queue)
    
    def empty(self):
        self.queue.clear()
        self.counter = 0.0
    
    def check(self):
        if len(self.queue) > 4.0:
            last_three = list(islice(self.queue, len(self.queue) - 4, len(self.queue) - 1))
            if all(self.queue[-1] < x for x in last_three):
                self.counter += 1
    

if __name__ == "__main__":
    q = CustomQueue(5)

    q.push(1)
    q.push(2)
    q.push(3)
    q.push(4)

    print(q.queue[-4])

