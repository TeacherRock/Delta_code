import numpy as np
import torch

from utils.motor_matlab import *
from utils.meta_RL_network_v2 import *
from utils.env import Environment, matlab_engine
from utils.save_data import *

import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
import ast
import numpy as np
import os

from PIL import Image
import torchvision.transforms as T

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

env = Environment(9, 3)
env.training = False

action_dim = env.action_dim
state_dim = env.state_dim

acc = False
model_type = "LSTM"  # "LSTM", "GRU", "Transformer"
# model_type = "GRU"  # "LSTM", "GRU", "Transformer"
# model_type = "Transformer"  # "LSTM", "GRU", "Transformer"


def inference(folder, ini_param, save_folder='', sys_num=0):

    os.makedirs(f"./record/{save_folder}", exist_ok=True)

    weight = f"./record/{folder}/weights/last_policy.pth"
    policy_net = torch.load(weight, weights_only=False)
    
    env.change_system_param(sys_num=sys_num)

    state = env.reset(ini_parameter=ini_param)

    max_steps = 30
    controller_parrams = []
    controller_parrams.append(env.controller_param.copy())

    with open(f"./record/{save_folder}" + "/reward.txt", "a", encoding="utf-8") as file:
        file.write(str(env.current_reward) + "\n")

    with open(f"./record/{save_folder}" + "/param.txt", "a", encoding="utf-8") as file:
        file.write(str(env.controller_param.copy()) + "\n")

    with open(
        f"./record/{save_folder}" + "/reward_info.txt", "a", encoding="utf-8") as file:
        file.write(str(env.reward_info.copy()) + "\n")

    return_step_idx = None
    return_terminal_condition_type = None

    # Initial Response
    if model_type in ["LSTM", "GRU"]:
        actor_hidden = policy_net.init_hidden(batch_size=1)

    current_dir = os.path.dirname(__file__)
    image_path = os.path.join(current_dir, "..", "..", "..", "system_parameters", "image_data")
    image_path = os.path.abspath(image_path)

    sys_num = int(sys_num)
    mag_path = os.path.join(image_path, f"{sys_num}/sys{sys_num}_mag.jpg")
    phase_path = os.path.join(image_path, f"{sys_num}/sys{sys_num}_phase.jpg")

    size=(128, 128)

    # 轉換為灰階並resize
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor()  # 轉為 [C, H, W]，灰階為 [1, H, W]
    ])

    mag_img = transform(Image.open(mag_path).convert("L"))  # [1, H, W]
    phase_img = transform(Image.open(phase_path).convert("L"))  # [1, H, W]

    spectrum_tensor = torch.cat([mag_img, phase_img], dim=0)  # [2, H, W]
    image_spectrum = spectrum_tensor.unsqueeze(0).float()
        
    for step_idx in range(max_steps):
        if model_type in ["LSTM", "GRU"]:
            action, actor_hidden = policy_net.select_action(state, image_spectrum, actor_hidden)
        elif model_type == "Transformer":
            action, _ = policy_net.select_action(state)
        # print(action)
        state, next_state, reward = env.step(state, action)
        state = next_state

        for key, value in env.reward_info.items():
            if isinstance(value, float) and math.isnan(value):
                env.reward_info[key] = 0
                
        with open(f"./record/{save_folder}"+"/reward.txt", "a", encoding="utf-8") as file:
            file.write(str(env.current_reward) + '\n')
        with open(f"./record/{save_folder}"+"/param.txt", "a", encoding="utf-8") as file:
            file.write(str(env.controller_param.copy()) + '\n')
        with open(f"./record/{save_folder}"+"/reward_info.txt", "a", encoding="utf-8") as file:
            file.write(str(env.reward_info.copy()) + '\n')

        if env.check_terminal_condition() and return_terminal_condition_type is None:
            return_step_idx = step_idx
            return_terminal_condition_type = env.terminal_condition_type
            
            performance(ini_param, env.controller_param.copy(), save_folder, sys_num)
    
    if return_terminal_condition_type is None:
        performance(ini_param, env.controller_param.copy(), save_folder, sys_num)
        
    return return_step_idx, return_terminal_condition_type
        
def draw_inference(folder, terminal_step_idx=None, terminal_type=""):
    overshoot_reward = []
    settling_reward = []
    # GM_position_reward = []
    GM_velocity_reward = []
    torque_reward = []
    reward = []
    overshoot_list = []
    settling_list = []
    # GM_position_list = []
    GM_velocity_list = []
    gain = []

    # 開啟文件並讀取每一行的第 9 列，將它們存入列表
    with open(f"./record/{folder}/reward.txt", 'r') as file:
        for line in file:
            columns = line.strip('[]\n').split(', ')  # 移除行首尾的 [] 和換行符，並用逗號分隔
            if len(columns) >= 9:  # 確保有至少 9 個欄位
                overshoot_reward.append(float(columns[0]))  # 將第 9 列的值轉為浮點數並添加到列表
                settling_reward.append(float(columns[1]))
                torque_reward.append(float(columns[7]))
                reward.append(float(columns[8]))
                try:
                    GM_velocity_reward.append(float(columns[9]))
                except:
                    pass
                # GM_velocity_reward.append(float(columns[10]))

    with open(f"./record/{folder}/reward_info.txt", 'r') as file:
        for line in file:
            line = line.strip().replace('nan', 'None')
            data = ast.literal_eval(line.strip())  # 將每行字串轉換為字典
            overshoot_list.append(data['overshoot'])
            settling_list.append(data['settling_time'])
            try:
                GM_velocity_list.append(data['GM_velocity'])
            except:
                pass

    with open(f"./record/{folder}/param.txt", "r") as file:
        for line in file:
            # 移除行首尾的括號和換行符，並轉換為數值
            row = np.fromstring(line.strip("[]\n"), sep=" ")
            gain.append(row)
    gain = np.array(gain)

    Kpp = gain[:, 0]
    Kvp = gain[:, 1]
    # Kvi = gain[:, 2]

    x = range(0, len(reward))
    # 繪製列表中的數據
    plt.plot(x, overshoot_reward, marker='o', label='overshoot')
    plt.plot(x, settling_reward, marker='o', label='settling')
    plt.plot(x, torque_reward, marker='o', label='torque')
    plt.plot(x, reward, marker='o', label='total reward')
    if terminal_step_idx is not None:
        plt.axvline(x=terminal_step_idx, color='red', linestyle='--', label=terminal_type)
    if GM_velocity_reward:
        # plt.plot(x, GM_position_reward, marker='o', label='GM_position')
        plt.plot(x, GM_velocity_reward, marker='o', label='GM_velocity')
    plt.xlabel("steps")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(f"./record/{folder}/reward.svg")
    plt.savefig(f"./record/{folder}/reward.jpg")
    plt.clf()

    plt.figure()
    plt.plot(Kpp, marker='o')
    if terminal_step_idx is not None:
        plt.axvline(x=terminal_step_idx, color='red', linestyle='--', label=terminal_type)
    plt.xlabel("steps")
    plt.ylabel("Kpp")
    plt.legend()
    plt.savefig(f"./record/{folder}/Kpp.svg")
    plt.savefig(f"./record/{folder}/Kpp.jpg")
    plt.clf()

    plt.figure()
    plt.plot(Kvp, marker='o')
    if terminal_step_idx is not None:
        plt.axvline(x=terminal_step_idx, color='red', linestyle='--', label=terminal_type)
    plt.xlabel("steps")
    plt.ylabel("Kvp")
    plt.savefig(f"./record/{folder}/Kvp.svg")
    plt.savefig(f"./record/{folder}/Kvp.jpg")
    plt.clf()

    # plt.figure()
    # plt.plot(Kvi, marker='o')
    # plt.xlabel("steps")
    # plt.ylabel("Kvi")
    # plt.savefig(f"./record/{folder}/Kvi.svg")
    # plt.savefig(f"./record/{folder}/Kvi.jpg")

    fig, ax1 = plt.subplots()

    # Plot the left y-axis data
    ax1.plot(x, overshoot_list, color='blue', marker='o', label='Maximum Overshoot')
    ax1.set_xlabel("steps")
    ax1.set_ylabel('Maximum Overshoot (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(x, settling_list, color='red', marker='o', label='Settling Time')
    ax2.set_ylabel('Settling Time (s)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Show the plot
    fig.tight_layout()
    plt.savefig(f"./record/{folder}/real.svg")
    plt.savefig(f"./record/{folder}/real.jpg")
    plt.clf()

    if GM_velocity_list:
        plt.figure()
        plt.plot(GM_velocity_list)
        plt.xlabel("steps")
        plt.ylabel("GM_velocity")
        plt.savefig(f"./record/{folder}/real_GM_velocity.svg")
        plt.savefig(f"./record/{folder}/real_GM_velocity.jpg")
        plt.clf()

    # plt.show()

    file_path = f"./record/{folder}/param.txt"
    output_path = f"./record/{folder}/difference.txt" 
    with open(file_path, "r") as file:
        lines = file.readlines()

    data = np.array([list(map(float, line.strip().strip("[]").split())) for line in lines])

    differences = np.diff(data, axis=0)
    np.savetxt(output_path, differences, fmt="%.8f", header="Differences:", comments="")

def performance(ini_param, last_param, save_folder, sys_num=0):

    os.makedirs("./record/"+save_folder+"/response", exist_ok=True)
    folder = save_folder + "/response"

    BW_Current = 2000.0

    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "..", "..", "..", "system_parameters", "system_parameters_table.csv")
    csv_path = os.path.abspath(csv_path)
    sys_settings = pd.read_csv(csv_path)
    sys_settings = sys_settings.dropna()

    system_choose = sys_settings[sys_settings['sys_num'] == sys_num].iloc[0]
    
    J1  = system_choose['J1']
    J2  = system_choose['J2']
    K12 = system_choose['K12']
    C12 = system_choose['C12']
    J   = J1 + J2
    fc  = [system_choose['fc_min'], system_choose['fc_max']]
    fs  = [system_choose['fs_min'], system_choose['fs_max']]
    B_sys = system_choose['B_sys']

    V_max = 4200 * 2*math.pi / 60.0  # 4200 (rpm)
    torque_disturb = B_sys*V_max
    tau_max = 2.54 - torque_disturb # 2.54 (Nm)
    A_max = tau_max / J
    random_T_load = 0.1

    matlab_engine.workspace["BW_Current"] = BW_Current
    matlab_engine.workspace["J1"]    = J1
    matlab_engine.workspace["J2"]    = J2
    matlab_engine.workspace["C12"]   = C12
    matlab_engine.workspace["K12"]   = K12
    matlab_engine.workspace["J"]     = J
    matlab_engine.workspace["B_sys"] = B_sys
    matlab_engine.workspace["Vmax"] = V_max
    matlab_engine.workspace["Amax"] = A_max
    matlab_engine.workspace["fc"] = matlab.double(fc)
    matlab_engine.workspace["fs"] = matlab.double(fs)
    matlab_engine.workspace["v_fric"]  = 0.0
    matlab_engine.workspace["T_load"] = random_T_load

    ## Initial Performance
    motor_ini = PPI_matlab()
    motor_ini.Kpp = ini_param[0]
    motor_ini.Kvp = ini_param[1]
    motor_ini.Kvi = last_param[2]
    error_ini, pos_ini, pos_cmd_ini, torque_ini, vel_ini = motor_ini.controller_step(0.0, 40.0, matlab_engine)
    tt_ini = np.arange(0, len(pos_ini)*0.001, 0.001)

    ## Last Performance
    matlab_engine.workspace["BW_Current"] = BW_Current
    matlab_engine.workspace["J1"]    = J1
    matlab_engine.workspace["J2"]    = J2
    matlab_engine.workspace["C12"]   = C12
    matlab_engine.workspace["K12"]   = K12
    matlab_engine.workspace["J"]     = J
    matlab_engine.workspace["B_sys"] = B_sys
    matlab_engine.workspace["Vmax"] = V_max
    matlab_engine.workspace["Amax"] = A_max
    matlab_engine.workspace["fc"] = matlab.double(fc)
    matlab_engine.workspace["fs"] = matlab.double(fs)
    matlab_engine.workspace["v_fric"]  = 0.0
    matlab_engine.workspace["T_load"] = random_T_load


    motor_last = PPI_matlab()
    motor_last.Kpp = last_param[0]
    motor_last.Kvp = last_param[1]
    motor_last.Kvi = last_param[2]
    error_last, pos_last, pos_cmd_last, torque_last, vel_last = motor_last.controller_step(0.0, 40.0, matlab_engine)
    tt_last = np.arange(0, len(pos_last)*0.001, 0.001)

    plt.figure()
    plt.plot(tt_ini, pos_cmd_ini, label='pos cmd')
    plt.plot(tt_ini, pos_ini, label='pos before tuning')
    plt.plot(tt_last, pos_last, label='pos after tuning')
    plt.ylabel("position (rad)")
    plt.xlabel("time (s)")
    plt.title("tracking")
    plt.legend()
    plt.savefig(f"./record/{folder}" + '/response.png')
    plt.savefig(f"./record/{folder}" + '/response.svg')
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.plot(tt_ini, pos_cmd_ini, label='pos cmd')
    plt.plot(tt_ini, pos_ini, label='pos before tuning')
    plt.plot(tt_last, pos_last, label='pos after tuning')

    if acc == True:
        plt.scatter(tt_ini[motor_ini.overshoot_idx-1], pos_ini[motor_ini.overshoot_idx-1], s=10, zorder=3, c='k', label='overshoot before tuning')
        plt.text(tt_ini[200], pos_cmd_ini[200]-5, f"(Before) (1 + overshoot)\n{round(error_ini['overshoot'], 6)+100 if round(error_ini['overshoot'], 6) != 0.0 else 100.0}%")
        plt.scatter(tt_last[motor_last.overshoot_idx-1], pos_last[motor_last.overshoot_idx-1], s=10, zorder=3, c='r', label='overshoot after tuning')
        plt.text(tt_last[200], pos_cmd_last[200]-10, f"(After) (1 + overshoot)\n{round(error_last['overshoot'], 6)+100 if round(error_last['overshoot'], 6) != 0.0 else 100.0}%")

        plt.vlines(tt_ini[motor_ini.settling_idx-1], pos_cmd_ini[motor_ini.settling_idx-1]-5, pos_cmd_ini[motor_ini.settling_idx-1]+5, colors='r', label='settling time before tuning')
        plt.vlines(tt_last[motor_last.settling_idx-1], pos_cmd_last[motor_last.settling_idx-1]-5, pos_cmd_last[motor_last.settling_idx-1]+5, colors='b', label='settling time after tuning')
        plt.text(tt_ini[400], pos_cmd_ini[400]-5, f"(Before) settling time\n{round(error_ini['settling_time'], 6)}s")
        plt.text(tt_last[400], pos_cmd_last[400]-10, f"(After) settling time\n{round(error_last['settling_time'], 6)}s")

    else:
        plt.scatter(tt_ini[motor_ini.overshoot_idx-1], pos_ini[motor_ini.overshoot_idx-1], s=10, zorder=3, c='k', label='overshoot before tuning')
        plt.text(tt_ini[100], pos_cmd_ini[100]-0.5, f"(Before) (1 + overshoot)\n{round(error_ini['overshoot'], 6)+100 if round(error_ini['overshoot'], 6) != 0.0 else 100.0}%")
        plt.scatter(tt_last[motor_last.overshoot_idx-1], pos_last[motor_last.overshoot_idx-1], s=10, zorder=3, c='r', label='overshoot after tuning')
        plt.text(tt_last[100], pos_cmd_last[100]-0.75, f"(After) (1 + overshoot)\n{round(error_last['overshoot'], 6)+100 if round(error_last['overshoot'], 6) != 0.0 else 100.0}%")

        plt.vlines(tt_ini[motor_ini.settling_idx-1], pos_cmd_ini[motor_ini.settling_idx-1]-0.1, pos_cmd_ini[motor_ini.settling_idx-1]+0.1, colors='r', label='settling time before tuning')
        plt.vlines(tt_last[motor_last.settling_idx-1], pos_cmd_last[motor_last.settling_idx-1]-0.1, pos_cmd_last[motor_last.settling_idx-1]+0.1, colors='b', label='settling time after tuning')
        plt.text(tt_ini[800], pos_cmd_ini[800]-0.5, f"(Before) settling time\n{round(error_ini['settling_time'], 6)}s")
        plt.text(tt_last[800], pos_cmd_last[800]-0.75, f"(After) settling time\n{round(error_last['settling_time'], 6)}s")

    plt.ylabel("position (rad)")
    plt.xlabel("time (s)")
    plt.title("tracking")
    plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f"./record/{folder}" + '/response_with_label.png')
    plt.savefig(f"./record/{folder}" + '/response_with_label.svg')
    plt.clf()

    plt.figure()
    plt.plot(tt_ini, np.array(pos_cmd_ini)-np.array(pos_ini), label='pos error before tuning')
    plt.plot(tt_last, np.array(pos_cmd_last)-np.array(pos_last), label='pos error after tuning')
    plt.ylabel("position error (rad)")
    plt.xlabel("time (s)")
    plt.title("position error")
    plt.legend()
    plt.savefig(f"./record/{folder}" + '/pos_error.png')
    plt.savefig(f"./record/{folder}" + '/pos_error.svg')
    plt.clf()

    plt.figure()
    plt.plot(tt_ini, vel_ini, label='velocity before tuning')
    plt.plot(tt_last, vel_last, label='velocity after tuning')
    plt.ylabel("velocity (rad/s)")
    plt.xlabel("time (s)")
    plt.title("velocity")
    plt.legend()
    plt.savefig(f"./record/{folder}" + '/velocity.png')
    plt.savefig(f"./record/{folder}" + '/velocity.svg')
    plt.clf()

    plt.figure()
    plt.plot(tt_ini, torque_ini, label='torque before tuning')
    plt.plot(tt_last, torque_last, label='torque after tuning')
    plt.ylabel("torque (Nm)")
    plt.xlabel("time (s)")
    plt.title("torque")
    plt.legend()
    plt.savefig(f"./record/{folder}" + '/torque.png')
    plt.savefig(f"./record/{folder}" + '/torque.svg')
    plt.clf()

if __name__ == "__main__":
    folder = '20241225_224419'
    inference(folder)