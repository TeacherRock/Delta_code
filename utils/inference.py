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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

env = Environment(13, 3)
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

    keys_to_remove = [
        "t_trap" "pos_trap", "vel_trap", "pos_error_trap",  "vel_error_trap",
        "pos_cmd_trap","vel_cmd_trap", "acc_cmd_trap", "torque_trap",
        "t_step", "pos_step", "vel_step", "pos_error_step" ,"pos_cmd_step",
        "torque_step"
    ]
    for key in keys_to_remove:
        env.reward_info.pop(key, None)

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
        for key in keys_to_remove:
            env.reward_info.pop(key, None)
                
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
    GM_velocity_reward = []
    torque_reward = []
    reward = []

    overshoot_list = []
    settling_list = []
    GM_velocity_list = [] 
    overshoot_load_list = []
    settling_load_list = []
    settling_001p_list = []
    settling_01p_list = []
    settling_1p_list = []
    settling_3p_list = []
    damping_ratio_list = []
    Ess_step_list = []
    Ess_trap_list = []

    gain = []

    # 開啟文件並讀取每一行的第 9 列，將它們存入列表
    with open(f"./record/{folder}/reward.txt", 'r') as file:
        for line in file:
            columns = line.strip('[]\n').split(', ')  # 移除行首尾的 [] 和換行符，並用逗號分隔
            if len(columns) >= 9:  # 確保有至少 9 個欄位
                overshoot_reward.append(float(columns[0]))  # 將第 9 列的值轉為浮點數並添加到列表
                settling_reward.append(float(columns[1]))

                torque_reward.append(float(columns[17]))
                try:
                    GM_velocity_reward.append(float(columns[18]))
                except:
                    pass
                reward.append(float(columns[-1]))

    with open(f"./record/{folder}/reward_info.txt", 'r') as file:

        # reward_names = [
        #     "overshoot", "overshoot_load", "settling_time", "settling_time_load", \
        #     "max_torque_flag", "GM_velocity",\
        #     "Emax_before", "Eavg_before", "Emax_after", "Eavg_after",\
        #     "settling_time_001p", "settling_time_01p", "settling_time_1p", "settling_time_3p", \
        #     "damping_ratio", "Ess_step", "Ess_trap"
        # ]
        for line in file:
            line = line.strip().replace('nan', 'None')
            line = line.strip().replace('inf', '-1.0')
            data = ast.literal_eval(line.strip())  # 將每行字串轉換為字典
            try:
                overshoot_list.append(data['overshoot'])
                settling_list.append(data['settling_time'])
                overshoot_load_list.append(data['overshoot_load'])
                settling_load_list.append(data['settling_time_load'])
                Ess_step_list.append(data['Ess_step'])
            except:
                print(f"KeyError step")

            try:
                GM_velocity_list.append(data['GM_velocity'])
            except:
                print(f"KeyError GM_velocity")
            
            try:
                settling_001p_list.append(data['settling_time_001p'])
                settling_01p_list.append(data['settling_time_01p'])
                settling_1p_list.append(data['settling_time_1p'])
                settling_3p_list.append(data['settling_time_3p'])
                damping_ratio_list.append(data['damping_ratio'])
                Ess_trap_list.append(data['Ess_trap'])
            except:
                print(f"KeyError trap")

    with open(f"./record/{folder}/param.txt", "r") as file:
        for line in file:
            # 移除行首尾的括號和換行符，並轉換為數值
            row = np.fromstring(line.strip("[]\n"), sep=" ")
            gain.append(row)
    
    gain = np.array(gain)
    Kpp = gain[:, 0]
    Kvp = gain[:, 1]
    Kvi = gain[:, 2]

    ## Plot reward ##
    x = range(0, len(reward))
    plt.plot(x, overshoot_reward, marker='o', label='overshoot')
    plt.plot(x, settling_reward, marker='o', label='settling')
    plt.plot(x, torque_reward, marker='o', label='torque')
    plt.plot(x, reward, marker='o', label='total reward')
    if terminal_step_idx is not None:
        plt.axvline(x=terminal_step_idx, color='red', linestyle='--', label=terminal_type)
    if GM_velocity_reward:
        plt.plot(x, GM_velocity_reward, marker='o', label='GM_velocity')
    plt.xlabel("steps")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(f"./record/{folder}/reward.svg")
    plt.savefig(f"./record/{folder}/reward.jpg")
    plt.clf()

    ## Plot gain ##
    def plot_and_save_param_curve(param_values, param_name, folder, terminal_step_idx=None, terminal_type="terminal"):
        plt.figure()
        plt.plot(param_values, marker='o')
        if terminal_step_idx is not None:
            plt.axvline(x=terminal_step_idx, color='red', linestyle='--', label=terminal_type)
            plt.legend()
        plt.xlabel("steps")
        plt.ylabel(param_name)
        os.makedirs(f"./record/{folder}", exist_ok=True)
        plt.savefig(f"./record/{folder}/{param_name}.svg")
        plt.savefig(f"./record/{folder}/{param_name}.jpg")
        plt.clf()
    
    plot_and_save_param_curve(Kpp, "Kpp", folder, terminal_step_idx, terminal_type)
    plot_and_save_param_curve(Kvp, "Kvp", folder, terminal_step_idx, terminal_type)
    plot_and_save_param_curve(Kvi, "Kvi", folder, terminal_step_idx, terminal_type)


    ## Plot matrics ##
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
    fig.tight_layout()
    plt.savefig(f"./record/{folder}/real.svg")
    plt.savefig(f"./record/{folder}/real.jpg")
    plt.clf()

    def plot_and_save_metric(metric_list, metric_name, folder):
        plt.figure()
        plt.plot(metric_list, marker='o')
        plt.xlabel("steps")
        plt.ylabel(metric_name)
        os.makedirs(f"./record/{folder}", exist_ok=True)
        plt.savefig(f"./record/{folder}/{metric_name}.svg")
        plt.savefig(f"./record/{folder}/{metric_name}.jpg")
        plt.clf()
    
    metrics_to_plot = [
        (overshoot_list, "overshoot"),
        (settling_list, "settling_time"),
        (overshoot_load_list, "overshoot_load"),
        (settling_load_list, "settling_time_load"),
        (Ess_step_list, "Ess_step"),
        (GM_velocity_list, "GM_velocity"),
        (settling_001p_list, "settling_time_001p"),
        (settling_01p_list, "settling_time_01p"),
        (settling_1p_list, "settling_time_1p"),
        (settling_3p_list, "settling_time_3p"),
        (damping_ratio_list, "damping_ratio"),
        (Ess_trap_list, "Ess_trap"),
    ]

    for metric_list, metric_name in metrics_to_plot:
        if len(metric_list) > 0:
            plot_and_save_metric(metric_list, metric_name, folder)
        else:
            print(f"[Warning] No data found for {metric_name}")


    file_path = f"./record/{folder}/param.txt"
    output_path = f"./record/{folder}/difference.txt" 
    with open(file_path, "r") as file:
        lines = file.readlines()

    data = np.array([list(map(float, line.strip().strip("[]").split())) for line in lines])
    differences = np.diff(data, axis=0)
    np.savetxt(output_path, differences, fmt="%.8f", header="Differences:", comments="")

    def plot_and_save_diff(diff_array, param_names, folder):
        for i, param_name in enumerate(param_names):
            plt.figure()
            plt.plot(diff_array[:, i], marker='o')
            plt.xlabel("steps")
            plt.ylabel(f"Δ{param_name}")
            plt.title(f"Difference of {param_name}")
            plt.savefig(f"./record/{folder}/diff_{param_name}.svg")
            plt.savefig(f"./record/{folder}/diff_{param_name}.jpg")
            plt.clf()

    param_names = ["Kpp", "Kvp", "Kvi"]
    plot_and_save_diff(differences, param_names, folder)

def performance(ini_param, last_param, save_folder, sys_num=0):

    def plot_text_table_comparison(dict_before, dict_after, save_path=None, cmd_type="step", title="Performance Comparison"):
        
        if cmd_type == "step":
            keys = [
                "overshoot", "settling_time", "max_torque_flag", "GM_velocity", "Ess_step"]
        elif cmd_type == "trap":
            keys = [
                "settling_time_001p", "settling_time_01p", "settling_time_1p", "settling_time_3p",
                "damping_ratio", "Ess_trap"
            ]

        data = [[key, f"{dict_before[key]:.4f}", f"{dict_after[key]:.4f}"] for key in keys]

        col_labels = ["Metric", "Before Tuning", "After Tuning"]
        cell_text = data

        plt.figure(figsize=(8, 0.4 * len(data) + 2))
        plt.axis('off')
        table = plt.table(cellText=cell_text,
                        colLabels=col_labels,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.3, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        plt.title(title, fontsize=12)
        plt.savefig(save_path+f"_{cmd_type}.svg", dpi=150, bbox_inches='tight')
        plt.savefig(save_path+f"_{cmd_type}.jpg", dpi=150, bbox_inches='tight')
        plt.close()

    os.makedirs("./record/"+save_folder+"/response", exist_ok=True)
    folder = save_folder + "/response"

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

    BW_Current = 2000.0
    V_max = 2000 * 2*math.pi / 60.0  # 2000 (rpm)
    torque_disturb = B_sys*V_max
    tau_max = 2.54 - torque_disturb # 2.54 (Nm)
    A_max = tau_max / J
    command_distance = 0.1 * V_max + V_max**2 / A_max # assmue t2 = 0.1s
    start_position = 0.0
    goal_position  = command_distance - start_position
    random_T_load = 0.5


    def get_performance(param, matlab_engine, cmd_type="step"):
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

        motor = PPI_matlab()
        motor.Kpp = param[0]
        motor.Kvp = param[1]
        motor.Kvi = param[2]
        if cmd_type == "step":
            error, pos, pos_cmd, torque, vel = motor.controller_step(0.0, 1.0, matlab_engine)
        else:
            error, pos, pos_cmd, torque, vel = motor.controller_trap(0.0, goal_position,matlab_engine)
        tt = np.arange(0, len(pos)*0.001, 0.001)
        return motor, error, pos, pos_cmd, torque, vel, tt

    motor_step_ini,  error_step_ini,  pos_step_ini,  pos_cmd_step_ini,  torque_step_ini,  vel_step_ini,  tt_step_ini  = \
        get_performance(ini_param,  matlab_engine, "step")

    motor_step_last, error_step_last, pos_step_last, pos_cmd_step_last, torque_step_last, vel_step_last, tt_step_last = \
        get_performance(last_param, matlab_engine, "step")
    
    # Trap Response
    motor_trap_ini,  error_trap_ini,  pos_trap_ini,  pos_trap_cmd_trap_ini,  torque_trap_ini,  vel_trap_ini,  tt_trap_ini  = \
        get_performance(ini_param,  matlab_engine, "trap")

    motor_trap_last, error_trap_last, pos_trap_last, pos_trap_cmd_trap_last, torque_trap_last, vel_trap_last, tt_trap_last = \
        get_performance(last_param, matlab_engine, "trap")


    plot_text_table_comparison(error_step_ini, error_step_last,
                               save_path=f"./record/{folder}/performance_comparison",
                               cmd_type="step",
                               title="Performance Comparison")
    
    plot_text_table_comparison(error_trap_ini, error_trap_last,
                               save_path=f"./record/{folder}/performance_comparison",
                               cmd_type="trap",
                               title="Performance Comparison")

    def plot_response_comparison(folder,
                tt_ini, pos_ini, pos_cmd_ini,
                tt_last, pos_last, pos_cmd_last,
                error_ini, error_last,
                motor_ini, motor_last,
                vel_ini, vel_last,
                torque_ini, torque_last, cmd_type):
        
        to_align = [
            tt_ini, pos_ini, pos_cmd_ini,
            tt_last, pos_last, pos_cmd_last,
            vel_ini, vel_last,
            torque_ini, torque_last
        ]

        # 取得最小長度
        min_len = min(len(arr) for arr in to_align)

        # 逐一裁切
        tt_ini = tt_ini[:min_len]
        pos_ini = pos_ini[:min_len]
        pos_cmd_ini = pos_cmd_ini[:min_len]
        tt_last = tt_last[:min_len]
        pos_last = pos_last[:min_len]
        pos_cmd_last = pos_cmd_last[:min_len]
        vel_ini = vel_ini[:min_len]
        vel_last = vel_last[:min_len]
        torque_ini = torque_ini[:min_len]
        torque_last = torque_last[:min_len]

        save_path = f"./record/{folder}/response_" + cmd_type
        os.makedirs(save_path, exist_ok=True)

        # --- [1] Tracking Response ---
        plt.figure()
        plt.plot(tt_ini, pos_cmd_ini, label='pos cmd')
        plt.plot(tt_ini, pos_ini, label='pos before tuning')
        plt.plot(tt_last, pos_last, label='pos after tuning')
        plt.ylabel("position (rad)")
        plt.xlabel("time (s)")
        plt.title(f"Tracking Response ({cmd_type})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}/tracking.png")
        plt.savefig(f"{save_path}/tracking.svg")
        plt.close()

        # --- [2] Tracking + Overshoot & Settling Time Labels ---
        plt.figure(figsize=(10, 6))
        plt.plot(tt_ini, pos_cmd_ini, label='pos cmd')
        plt.plot(tt_ini, pos_ini, label='pos before tuning')
        plt.plot(tt_last, pos_last, label='pos after tuning')
        # overshoot
        if cmd_type == "step":
            plt.scatter(tt_ini[motor_ini.overshoot_idx-1], pos_ini[motor_ini.overshoot_idx-1],
                        s=10, zorder=3, c='k', label='overshoot before')
            plt.text(tt_ini[100], pos_cmd_ini[100]-0.5,
                    f"(Before) (1+overshoot)\n{round(error_ini['overshoot'], 6)+100:.2f}%")

            plt.scatter(tt_last[motor_last.overshoot_idx-1], pos_last[motor_last.overshoot_idx-1],
                        s=10, zorder=3, c='r', label='overshoot after')
            plt.text(tt_last[100], pos_cmd_last[100]-0.75,
                    f"(After) (1+overshoot)\n{round(error_last['overshoot'], 6)+100:.2f}%")

            # settling time
            plt.vlines(tt_ini[motor_ini.settling_idx-1],
                    pos_cmd_ini[motor_ini.settling_idx-1]-0.1,
                    pos_cmd_ini[motor_ini.settling_idx-1]+0.1,
                    colors='r', label='settling before')

            plt.vlines(tt_last[motor_last.settling_idx-1],
                    pos_cmd_last[motor_last.settling_idx-1]-0.1,
                    pos_cmd_last[motor_last.settling_idx-1]+0.1,
                    colors='b', label='settling after')

            plt.text(tt_ini[800], pos_cmd_ini[800]-0.5,
                    f"(Before) settling time\n{round(error_ini['settling_time'], 6)}s")
            plt.text(tt_last[800], pos_cmd_last[800]-0.75,
                    f"(After) settling time\n{round(error_last['settling_time'], 6)}s")
        elif cmd_type == "trap":
            pass

        plt.ylabel("position (rad)")
        plt.xlabel("time (s)")
        plt.title("Tracking(Step)")
        plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f"{save_path}/tracking_labeled_{cmd_type}.png")
        plt.savefig(f"{save_path}/tracking_labeled_{cmd_type}.svg")
        plt.close()

        # --- [3] Position Error ---
        plt.figure()
        plt.plot(tt_ini, np.array(pos_cmd_ini) - np.array(pos_ini), label='error before')
        plt.plot(tt_last, np.array(pos_cmd_last) - np.array(pos_last), label='error after')
        plt.ylabel("Position error (rad)")
        plt.xlabel("time (s)")
        plt.title(f"Position Error ({cmd_type})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}/pos_error_{cmd_type}.png")
        plt.savefig(f"{save_path}/pos_error_{cmd_type}.svg")
        plt.close()

        # --- [4] Velocity ---
        plt.figure()
        plt.plot(tt_ini, vel_ini, label='velocity before')
        plt.plot(tt_last, vel_last, label='velocity after')
        plt.ylabel("velocity (rad/s)")
        plt.xlabel("time (s)")
        plt.title(f"Velocity ({cmd_type})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}/velocity_{cmd_type}.png")
        plt.savefig(f"{save_path}/velocity_{cmd_type}.svg")
        plt.close()

        # --- [5] Torque ---
        plt.figure()
        plt.plot(tt_ini, torque_ini, label='torque before')
        plt.plot(tt_last, torque_last, label='torque after')
        plt.ylabel("torque (Nm)")
        plt.xlabel("time (s)")
        plt.title(f"Torque ({cmd_type})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}/torque_{cmd_type}.png")
        plt.savefig(f"{save_path}/torque_{cmd_type}.svg")
        plt.close()

    plot_response_comparison(folder,
                tt_step_ini, pos_step_ini, pos_cmd_step_ini,
                tt_step_last, pos_step_last, pos_cmd_step_last,
                error_step_ini, error_step_last,
                motor_step_ini, motor_step_last,
                vel_step_ini, vel_step_last,
                torque_step_ini, torque_step_last, "step")
    plot_response_comparison(folder,
                tt_trap_ini, pos_trap_ini, pos_trap_cmd_trap_ini,
                tt_trap_last, pos_trap_last, pos_trap_cmd_trap_last,
                error_trap_ini, error_trap_last,
                motor_trap_ini, motor_trap_last,
                vel_trap_ini, vel_trap_last,
                torque_trap_ini, torque_trap_last, "trap")
    
    def plot_T_load_results(folder, error_info, ctrl_param, cmd_type, before_or_after="before"):

        Kpp, Kvp, Kvi = ctrl_param

        save_path = f"./record/{folder}/response_" + cmd_type

        if cmd_type == "step":
            textstr = '\n'.join([
                    f"overshoot          : {error_info['overshoot']:.8f}",
                    f"overshoot_load     : {error_info['overshoot_load']:.8f}",
                    f"settling_time      : {error_info['settling_time']:.8f}",
                    f"settling_time_load : {error_info['settling_time_load']:.8f}",
                    f"GM_velocity        : {error_info['GM_velocity']:.8f}",
                    f"Ess_step           : {error_info['Ess_step']:.8f}",
                ])
            t = np.arange(len(error_info["pos_step"])) * 0.001
            pos = error_info["pos_step"]
            pos_cmd = error_info["pos_cmd_step"]
            torque  = error_info["torque_step"]

        elif cmd_type == "trap":
            textstr = '\n'.join([
                    f"settling_time_001p : {error_info['settling_time_001p']:.8f}",
                    f"settling_time_01p  : {error_info['settling_time_01p']:.8f}",
                    f"settling_time_1p   : {error_info['settling_time_1p']:.8f}",
                    f"settling_time_3p   : {error_info['settling_time_3p']:.8f}",
                    f"damping_ratio      : {error_info['damping_ratio']:.8f}",
                    f"Ess_trap           : {error_info['Ess_trap']:.8f}", 
                ])
            
            t = np.arange(len(error_info["pos_trap"])) * 0.001
            pos = error_info["pos_trap"]
            pos_cmd = error_info["pos_cmd_trap"]
            torque  = error_info["torque_trap"]


        plt.figure(figsize=(10, 6))
        # 主圖
        ax = plt.gca()
        ax.plot(t, pos, label='pos')
        ax.plot(t, pos_cmd, label='pos_cmd')
        ax.plot(t[-600], pos_cmd[-600], label='Add T_Load', marker='o', markersize=3, color='r')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (rad)")
        ax.set_title(f"Position: Kpp={Kpp}, Kvp={Kvp}, Kvi={Kvi}")
        ax.legend(loc="best")

        # 文字方塊
        ax.text(1.02, 0.45, "Sim:\n" + textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.7))

        # ===== 放大區塊 =====
        # 放大時間與位置範圍（可依你畫框調整）
        x1, x2 = t[-1]-0.6, t[-1]+0.05
        y1 = max(pos_cmd)-0.05 if max(pos_cmd) < max(pos) else max(pos)-0.05
        y2 = max(pos_cmd)+0.05 if max(pos_cmd) > max(pos) else max(pos)+0.05

        # 建立小圖
        axins = inset_axes(ax, width="50%", height="50%", loc="center")
        axins.plot(t, pos, label='pos')
        axins.plot(t, pos_cmd, label='pos_cmd')
        axins.plot(t[-600], pos_cmd[-600], label='Add T_Load', marker='o', markersize=3, color='r')
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xlabel("Time (s)", fontsize=8)   # 或者省略文字只留下刻度
        axins.set_ylabel("Position (rad)", fontsize=8)
        axins.tick_params(axis='both', which='both', labelsize=7)
        axins.grid(True)

        # 畫框線連接主圖與小圖
        mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="blue", linewidth=1.2)

        plt.tight_layout()

        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path+f"/pos_{before_or_after}.png", dpi=150, bbox_inches='tight')
        plt.savefig(save_path+f"/pos_{before_or_after}.svg", dpi=150, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(t, torque)
        plt.xlabel("Time (s)")
        plt.ylabel("Torque (Nm)")
        plt.title(f"Torque: Kpp={Kpp}, Kvp={Kvp}, Kvi={Kvi}")
        plt.tight_layout()
        plt.savefig(save_path+f"/tor_{before_or_after}que.png", dpi=150, bbox_inches='tight')
        plt.savefig(save_path+f"/tor_{before_or_after}que.svg", dpi=150, bbox_inches='tight')
        plt.close()
    
    plot_T_load_results(folder, error_step_ini, ini_param, "step", "before")
    plot_T_load_results(folder, error_step_last, last_param, "step", "after")
    plot_T_load_results(folder, error_trap_ini, ini_param, "trap", "before")
    plot_T_load_results(folder, error_trap_last, last_param, "trap", "after")


if __name__ == "__main__":
    folder = '20241225_224419'
    inference(folder)