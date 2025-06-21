import numpy as np
import os

import numpy as np
import torch

from utils.motor_matlab import *
from utils.RL_network import *
from utils.env import Environment, matlab_engine
from utils.save_data import *

import matplotlib.pyplot as plt
import ast
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

env = Environment(11, 2)
env.training = False

action_dim = env.action_dim
state_dim = env.state_dim
hidden_dim = 512

acc = False

sys_num = 3

def inference(folder, ini_param, save_folder='', sys_num=0):

    weight = f"./record/{folder}/weights/last_policy.pth"
    policy_net = torch.load(weight)

    env.change_J_and_B(sys_num=sys_num)

    state = env.reset(ini_parameter=ini_param)

    max_steps = 30
    controller_parrams = []
    controller_parrams.append(env.controller_param.copy())

    return_step_idx = None
    return_terminal_condition_type = None


    Jm_list = [] # Motor Inertia
    JL_list = [] # Load Inertia
    Ks_list = [] # Spring Coefficient
    Cs_list = [] # Damping Coefficient

    Kpp_list = [] # Position loop proportional Gain
    Kvp_list = [] # Velocity loop proportional Gain

    overshoot_list = [] # Overshoot
    settling_time_list = [] # Settling Time
    GM_list = [] # Gain Margin

    dKpp_list = [] # Tuning of position loop proportional Gain
    dKvp_list = [] # Tuning of velocity loop proportional Gain

    record_dic = {
        "Jm": Jm_list,
        "JL": JL_list,
        "Ks": Ks_list,
        "Cs": Cs_list,
        "Kpp": Kpp_list,
        "Kvp": Kvp_list,
        "overshoot": overshoot_list,
        "settling_time": settling_time_list,
        "GM": GM_list,
        "dKpp": dKpp_list,
        "dKvp": dKvp_list
    }


    # Initial Response
    for step_idx in range(max_steps):
        action = policy_net.get_action(state)

        Jm_list.append(env.J1)
        JL_list.append(env.J2)
        Ks_list.append(env.K12)
        Cs_list.append(env.C12)
        Kpp_list.append(env.controller_param[0])
        Kvp_list.append(env.controller_param[1])
        overshoot_list.append(env.reward_info["overshoot"])
        settling_time_list.append(env.reward_info["settling_time"])
        GM_list.append(env.reward_info["GM_velocity"])

        state, next_state, reward = env.step(state, action)
        state = next_state

        for key, value in env.reward_info.items():
            if isinstance(value, float) and math.isnan(value):
                env.reward_info[key] = 0

        if env.check_terminal_condition() and return_terminal_condition_type is None:
            return_step_idx = step_idx
            return_terminal_condition_type = env.terminal_condition_type

        dKpp_list.append(env._action[0])
        dKvp_list.append(env._action[1])

    df = pd.DataFrame(record_dic)
    df.to_csv(save_folder+".csv", index=False)


def generate_dataset(root_folder="./record"):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        if os.path.isdir(folder_path) and folder_name.startswith("20250401_005300"):
            os.makedirs("./record/" + folder_name + "/dataset_v7/", exist_ok=True)
            print(f"Processing folder: {folder_name}")

            # Kpp_ini_params = [np.array([150.0]), np.array([200.0]),  np.array([500.0]),
            #                   np.array([900.0])]
            
            # Kvp_ini_params = [np.array([300.0]),
            #                   np.array([500.0]), np.array([700.0]), np.array([900.0])]

            # Kpp_ini_params = [np.array([150.0]), np.array([200.0]),  np.array([250.0]),  np.array([300.0]),  np.array([350.0]),
            #                   np.array([400.0]), np.array([450.0]),  np.array([500.0]),  np.array([550.0]),  np.array([600.0]),  np.array([650.0]),
            #                   np.array([700.0]), np.array([750.0]),  np.array([800.0]),  np.array([850.0]),  np.array([900.0])]
            
            # Kvp_ini_params = [np.array([300.0]), np.array([400.0]), np.array([500.0]), np.array([600.0]), np.array([700.0]), np.array([800.0]), np.array([900.0])]

            Kpp_ini_params = [np.array([float(kpp)]) for kpp in range(150, 960, 20)]
            Kvp_ini_params = [np.array([float(kvp)]) for kvp in range(150, 960, 20)]
            
            for Kpp in Kpp_ini_params:
                for Kvp in Kvp_ini_params:
                    print("/Kpp_" + str(int(Kpp[0])) + "_Kvp_"  + str(int(Kvp[0])))
                    save_folder = "./record/" + folder_name + "/dataset_v7/" + "/Kpp_" + str(int(Kpp[0])) + "_Kvp_" +str(int(Kvp[0])) 
                    inference(folder_name, ini_param=[Kpp, Kvp], save_folder=save_folder, sys_num=sys_num)

if __name__ == "__main__":
    generate_dataset()