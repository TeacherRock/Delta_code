import matlab.engine
import matplotlib.pyplot as plt
import time
import numpy as np
import math

class PPI_matlab:
    def __init__(self):
        self.Kpp = 0
        self.Kpf = 0
        self.Kvp = 0
        self.Kvi = 0
        self.Kvf = 0
        # self.sampT = 1 / 16000
        self.sampT = 1 / 1000

    def controller_step(self, start_position, goal_position, eng, file_name=None):
        eng.workspace["Kpp"] = self.Kpp
        eng.workspace["Kvp"] = self.Kvp
        eng.workspace["Kvi"] = self.Kvi
        eng.workspace["sampT"] = self.sampT

        eng.workspace["start_position"] = start_position
        eng.workspace["goal_position"]  = goal_position

        # run matlab
        if file_name is None:
            eng.run("matlab_module/simulation_no_feedforward_step.m", nargout=0)
        else:
            eng.run("matlab_module/" + file_name, nargout=0)

        tt = np.array(eng.eval("tout", nargout=1)).T.tolist()[0]
        pos = np.array(eng.eval("pos", nargout=1)).T.tolist()[0]
        vel = np.array(eng.eval("vel", nargout=1)).T.tolist()[0]
        pos_error = np.array(eng.eval("pos_error", nargout=1)).T.tolist()[0]
        pos_cmd = np.array(eng.eval("pos_cmd", nargout=1)).T.tolist()[0]
        torque = np.array(eng.eval("torque", nargout=1)).T.tolist()[0]
        GM_velocity = eng.eval("Gm_velocity", nargout=1)

        sim_data = {
            "t_step"           : tt,
            "pos_step"         : pos,
            "vel_step"         : vel, 
            "pos_error_step"   : pos_error, 
            "pos_cmd_step"     : pos_cmd,
            "torque_step"      : torque,
            "GM_velocity"      : GM_velocity
        }

        # clear matlab
        eng.eval("clear", nargout=0)
        eng.eval("clc", nargout=0)

        error = self.calculate_reward_info_step(sim_data)

        sim_data.update(error)

        return sim_data, pos, pos_cmd, torque, vel
    
    def controller_trap(self, start_position, goal_position, eng, file_name=None):
        eng.workspace["Kpp"] = self.Kpp
        eng.workspace["Kvp"] = self.Kvp
        eng.workspace["Kvi"] = self.Kvi
        eng.workspace["sampT"] = self.sampT

        eng.workspace["start_position"] = start_position
        eng.workspace["goal_position"]  = goal_position

        # run matlab
        if file_name is None:
            eng.run("matlab_module/simulation_no_feedforward_trap.m", nargout=0)
        else:
            eng.run("matlab_module/" + file_name, nargout=0)

        tt = np.array(eng.eval("tout", nargout=1)).T.tolist()[0]
        pos = np.array(eng.eval("pos", nargout=1)).T.tolist()[0]
        vel = np.array(eng.eval("vel", nargout=1)).T.tolist()[0]
        pos_error = np.array(eng.eval("pos_error", nargout=1)).T.tolist()[0]
        vel_error = np.array(eng.eval("vel_error", nargout=1)).T.tolist()[0]
        pos_cmd = np.array(eng.eval("pos_cmd", nargout=1)).T.tolist()[0]
        vel_cmd = np.array(eng.eval("vel_cmd", nargout=1)).T.tolist()[0]
        acc_cmd = np.array(eng.eval("acc_cmd", nargout=1)).T.tolist()[0]
        torque = np.array(eng.eval("torque", nargout=1)).T.tolist()[0]

        sim_data = {
            "t_trap"         : tt,
            "pos_trap"       : pos,
            "vel_trap"       : vel, 
            "pos_error_trap" : pos_error, 
            "vel_error_trap" : vel_error,
            "pos_cmd_trap"   : pos_cmd,
            "vel_cmd_trap"   : vel_cmd,
            "acc_cmd_trap"   : acc_cmd,
            "torque_trap"    : torque,
        }

        # clear matlab
        eng.eval("clear", nargout=0)
        eng.eval("clc", nargout=0)

        error = self.calculate_reward_info_trap(sim_data)

        sim_data.update(error)

        return sim_data, pos, pos_cmd, torque, vel
  
    def calculate_reward_info_step(self, sim_data):
        pos = np.array(sim_data["pos_step"])[:-1500]
        pos_steady = np.array(sim_data["pos_step"])[-1500:]
        vel = sim_data["vel_step"]
        pos_error = sim_data["pos_error_step"][:-1500]
        pos_cmd = sim_data["pos_cmd_step"][:-1500]
        pos_cmd_steady = sim_data["pos_cmd_step"][-1500:]
        torque  = sim_data["torque_step"]
        GM_velocity  = sim_data["GM_velocity"]

        pos_cmd = 1.0
        max_pos = max(pos)
        pos_idx = np.argmax(pos)
        overshoot = (max_pos - pos_cmd) / pos_cmd * 100 if max_pos - pos_cmd > 0.0 else 0.0
        self.overshoot_idx = pos_idx
        self.settling_idx = len(pos)

        max_pos_load = max(pos_steady)
        pos_load_idx = np.argmax(pos_steady)
        overshoot_load = (max_pos_load - pos_cmd) / pos_cmd * 100 if max_pos_load - pos_cmd > 0.0 else 0.0
        self.overshoot_load_idx = pos_load_idx
        self.settling_load_idx = len(pos_steady)

        def get_settling_time(error_band = 1e-6):
            steady_start_idx = 0
            lower_bound = pos_cmd - error_band
            upper_bound = pos_cmd + error_band
            settling_idx = len(pos)
            for i in range(steady_start_idx, len(pos)):
                if np.all(pos[i:] >= lower_bound) and np.all(pos[i:] <= upper_bound):
                    settling_idx = i
                    self.settling_idx = settling_idx
                    break
            settling_time = (settling_idx) * 0.001
            return settling_time
        
        def get_settling_time_load(error_band = 1e-6):
            steady_start_idx = 0
            lower_bound = pos_cmd - error_band
            upper_bound = pos_cmd + error_band
            settling_idx = len(pos_steady)
            for i in range(steady_start_idx, len(pos_steady)):
                if np.all(pos_steady[i:] >= lower_bound) and np.all(pos_steady[i:] <= upper_bound):
                    settling_idx = i
                    self.settling_load_idx = settling_idx
                    break
            settling_time = (settling_idx) * 0.001
            return settling_time
        
        settling_time = get_settling_time(0.03)
        settling_time_load = get_settling_time_load(0.003)

        # over max torque flag #
        max_torque = np.max(np.abs(torque))
        if max_torque > 4.4:
            max_torque_flag = max_torque - 4.4
        else:
            max_torque_flag = 0.0

        Ess_step = np.mean(np.abs(pos_steady[-20:] - 1.0))

        reward_info = {
            "overshoot"          : overshoot,
            "settling_time"      : settling_time,
            "max_torque_flag"    : max_torque_flag,
            "GM_velocity"        : GM_velocity,
            "Ess_step"           : Ess_step,
            "overshoot_load" : overshoot_load,
            "settling_time_load" : settling_time_load
        }

        return reward_info
    
    def calculate_reward_info_trap(self, sim_data):
        pos = np.array(sim_data["pos_trap"])[:-1500]
        pos_steady = np.array(sim_data["pos_trap"])[-1500:]
        vel = sim_data["vel_trap"]
        pos_error = sim_data["pos_error_trap"][:-1500]
        vel_error = sim_data["vel_error_trap"]
        pos_cmd = sim_data["pos_cmd_trap"][:-1500]
        pos_cmd_steady = sim_data["pos_cmd_trap"][-1500:]
        vel_cmd = sim_data["vel_cmd_trap"]
        acc_cmd = sim_data["acc_cmd_trap"]

        torque  = sim_data["torque_trap"]

        steady_start_idx = find_first_occurrence(pos_cmd, pos_cmd[-1])

        def get_settling_time(error_band = 1e-6):
            lower_bound = np.array(pos_cmd) - error_band * pos_cmd[-1]
            upper_bound = np.array(pos_cmd) + error_band * pos_cmd[-1]
            settling_idx = len(pos)
            for i in range(steady_start_idx, len(pos)):
                if np.all(pos[i:] >= lower_bound[i:]) and np.all(pos[i:] <= upper_bound[i:]):
                    settling_idx = i
                    break
            settling_time = (settling_idx - steady_start_idx) * 0.001
            return settling_time
        
        settling_time_001p = get_settling_time(1e-4)
        settling_time_01p  = get_settling_time(1e-3)
        settling_time_1p   = get_settling_time(1e-2)
        settling_time_3p   = get_settling_time(3e-2)

        Emax_before = np.max(np.abs(pos_error[:steady_start_idx]))
        Eavg_before = np.mean(np.square(pos_error[:steady_start_idx]))

        Emax_after = np.max(np.abs(pos_error[steady_start_idx:]))
        Eavg_after = np.mean(np.square(pos_error[steady_start_idx:]))

        Emax_after_more = np.max(np.abs(pos_error[steady_start_idx-20:]))
        Eavg_after_more = np.mean(np.square(pos_error[steady_start_idx-20:]))

        damping_ratio =  -np.log(Emax_after) / np.sqrt(np.pi**2+ np.log(Emax_after)**2)

        Ess_trap = np.mean(np.abs(pos_steady[-20:] - pos_cmd_steady[-20:]))

        reward_info = {
            "Emax_before"     : Emax_before,
            "Eavg_before"     : Eavg_before,
            "Emax_after"      : Emax_after,
            "Eavg_after"      : Eavg_after,
            "Emax_after_more" : Emax_after_more,
            "Eavg_after_more" : Eavg_after_more,
            "settling_time_001p" : settling_time_001p,
            "settling_time_01p"  : settling_time_01p,
            "settling_time_1p"   : settling_time_1p,
            "settling_time_3p"   : settling_time_3p,
            "damping_ratio" : damping_ratio,
            "Ess_trap" : Ess_trap,
            "Ess" : 0.0
        }

        return reward_info
    
    def double_to_list(self, lists):
        modified_lists = []
        for term in lists:
            modified_term = []
            for i in range(len(term)):
                modified_term.append(term[i][0])
            modified_lists.append(modified_term)

        return modified_lists

def find_first_occurrence(array, x):
    for i in range(len(array)):
        if array[i] == x:
            return i
    print("No maximum value")
    return -1


if __name__ == "__main__":
    matlab_engine = matlab.engine.start_matlab()
    motor = PPI_matlab()
    frequency = 16000
    sampT     = 1/frequency
    J_M = 0.45e-4
    J_L = 0.45e-4 * 9 
    J_base = J_M + J_L
    B_base = J_M / 10.0
    J = J_base * 1.01
    J_sys = J_base
    B_sys = B_base
    V_max = 4200 * 2*math.pi / 60.0  # 4200 (rpm)
    torque_disturb = B_sys*V_max
    tau_max = 2.54 - torque_disturb # 2.54 (Nm)
    A_max = tau_max / J_sys
    matlab_engine.workspace["J"]     = J
    matlab_engine.workspace["J_sys"] = J_sys
    matlab_engine.workspace["B_sys"] = B_sys
    matlab_engine.workspace["Vmax"] = V_max
    matlab_engine.workspace["Amax"] = A_max
    matlab_engine.workspace["sampT"] = sampT
    motor.controller(-40.0, 40.0, matlab_engine)
