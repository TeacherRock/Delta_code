from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.pyplot as plt
import matlab.engine
import pandas as pd
import numpy as np
import math
import os

current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, "..", "..", "system_parameters", "system_parameters_table.csv")
csv_path = os.path.abspath(csv_path)
sys_settings = pd.read_csv(csv_path)
sys_settings = sys_settings.dropna()

def find_first_occurrence(array, x):
    for i in range(len(array)):
        if array[i] == x:
            return i
    print("No maximum value")
    return -1

def calculate_reward_info_step(sim_data):
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

    max_pos_load = max(pos_steady)
    pos_load_idx = np.argmax(pos_steady)
    overshoot_load = (max_pos_load - pos_cmd) / pos_cmd * 100 if max_pos_load - pos_cmd > 0.0 else 0.0

    def get_settling_time(error_band = 1e-6):
        steady_start_idx = 0
        lower_bound = pos_cmd - error_band
        upper_bound = pos_cmd + error_band
        settling_idx = len(pos)
        for i in range(steady_start_idx, len(pos)):
            if np.all(pos[i:] >= lower_bound) and np.all(pos[i:] <= upper_bound):
                settling_idx = i
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
                break
        settling_time = (settling_idx) * 0.001
        return settling_time
    
    settling_time      = get_settling_time(0.03)
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
        "overshoot_load"     : overshoot_load,
        "settling_time_load" : settling_time_load
    }

    return reward_info

def calculate_reward_info_trap(sim_data):
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

def simulation_trap(Kpp=100.0, Kvp=200.0, Kvi=0.0):

    system_choose = sys_settings[sys_settings['sys_num'] == sys_num].iloc[0]
    J1  = system_choose['J1']
    J2  = system_choose['J2']
    K12 = system_choose['K12']
    C12 = system_choose['C12']
    war = system_choose['war']
    wr =  system_choose['wr']
    J   = J1 + J2
    fc  = [system_choose['fc_min'], system_choose['fc_max']]
    fs  = [system_choose['fs_min'], system_choose['fs_max']]
    B_sys = system_choose['B_sys']

    V_max = 2000 * 2*math.pi / 60.0  # 2000 (rpm)
    tau_max = 2.54 - B_sys*V_max
    A_max = tau_max / J

    command_distance = 0.1 * V_max + V_max**2 / A_max # assmue t2 = 0.1s
    start_position = 0.0
    goal_position  = command_distance - start_position

    BW_Current = 2000.0
    matlab_engine.workspace["BW_Current"] = BW_Current
    matlab_engine.workspace["J1"]    = J1
    matlab_engine.workspace["J2"]    = J2
    matlab_engine.workspace["C12"]   = C12
    matlab_engine.workspace["K12"]   = K12
    matlab_engine.workspace["J"]     = J
    matlab_engine.workspace["B_sys"] = B_sys
    matlab_engine.workspace["Vmax"]  = V_max
    matlab_engine.workspace["Amax"]  = A_max
    matlab_engine.workspace["fc"] = matlab.double(fc)
    matlab_engine.workspace["fs"] = matlab.double(fs)
    matlab_engine.workspace["v_fric"]  = 0.0

    matlab_engine.workspace["Kpp"] = Kpp
    matlab_engine.workspace["Kvp"] = Kvp
    matlab_engine.workspace["Kvi"] = Kvi
    matlab_engine.workspace["sampT"] = 0.001

    matlab_engine.workspace["T_load"] = T_load

    matlab_engine.workspace["start_position"] = start_position
    matlab_engine.workspace["goal_position"]  = goal_position

    # run matlab
    matlab_engine.run("matlab_module/simulation_no_feedforward_trap.m", nargout=0)

    tt = np.array(matlab_engine.eval("tout", nargout=1)).T.tolist()[0]
    pos = np.array(matlab_engine.eval("pos", nargout=1)).T.tolist()[0]
    vel = np.array(matlab_engine.eval("vel", nargout=1)).T.tolist()[0]
    pos_error = np.array(matlab_engine.eval("pos_error", nargout=1)).T.tolist()[0]
    vel_error = np.array(matlab_engine.eval("vel_error", nargout=1)).T.tolist()[0]
    pos_cmd = np.array(matlab_engine.eval("pos_cmd", nargout=1)).T.tolist()[0]
    vel_cmd = np.array(matlab_engine.eval("vel_cmd", nargout=1)).T.tolist()[0]
    acc_cmd = np.array(matlab_engine.eval("acc_cmd", nargout=1)).T.tolist()[0]
    torque = np.array(matlab_engine.eval("torque", nargout=1)).T.tolist()[0]

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
    matlab_engine.eval("clear", nargout=0)
    matlab_engine.eval("clc", nargout=0)

    error = calculate_reward_info_trap(sim_data)

    sim_data.update(error)

    return sim_data

def simulation_step(Kpp=100.0, Kvp=200.0, Kvi=0.0):

    system_choose = sys_settings[sys_settings['sys_num'] == sys_num].iloc[0]
    J1  = system_choose['J1']
    J2  = system_choose['J2']
    K12 = system_choose['K12']
    C12 = system_choose['C12']
    war = system_choose['war']
    wr =  system_choose['wr']
    J   = J1 + J2
    fc  = [system_choose['fc_min'], system_choose['fc_max']]
    fs  = [system_choose['fs_min'], system_choose['fs_max']]
    B_sys = system_choose['B_sys']

    V_max = 2000 * 2*math.pi / 60.0  # 2000 (rpm)
    tau_max = 2.54 - B_sys*V_max
    A_max = tau_max / J

    command_distance = 0.1 * V_max + V_max**2 / A_max # assmue t2 = 0.1s
    start_position = 0.0
    goal_position  = command_distance - start_position

    BW_Current = 2000.0
    matlab_engine.workspace["BW_Current"] = BW_Current
    matlab_engine.workspace["J1"]    = J1
    matlab_engine.workspace["J2"]    = J2
    matlab_engine.workspace["C12"]   = C12
    matlab_engine.workspace["K12"]   = K12
    matlab_engine.workspace["J"]     = J
    matlab_engine.workspace["B_sys"] = B_sys
    matlab_engine.workspace["Vmax"]  = V_max
    matlab_engine.workspace["Amax"]  = A_max
    matlab_engine.workspace["fc"] = matlab.double(fc)
    matlab_engine.workspace["fs"] = matlab.double(fs)
    matlab_engine.workspace["v_fric"]  = 0.0

    matlab_engine.workspace["Kpp"] = Kpp
    matlab_engine.workspace["Kvp"] = Kvp
    matlab_engine.workspace["Kvi"] = Kvi
    matlab_engine.workspace["sampT"] = 0.001
    
    matlab_engine.workspace["T_load"] = T_load

    matlab_engine.workspace["start_position"] = start_position
    matlab_engine.workspace["goal_position"]  = goal_position

    # run matlab
    # matlab_engine.run("matlab_module/simulation_no_feedforward_step.m", nargout=0)
    matlab_engine.run("matlab_module/simulation_no_feedforward_step_sat.m", nargout=0)

    tt = np.array(matlab_engine.eval("tout", nargout=1)).T.tolist()[0]
    pos = np.array(matlab_engine.eval("pos", nargout=1)).T.tolist()[0]
    vel = np.array(matlab_engine.eval("vel", nargout=1)).T.tolist()[0]
    pos_error = np.array(matlab_engine.eval("pos_error", nargout=1)).T.tolist()[0]
    pos_cmd = np.array(matlab_engine.eval("pos_cmd", nargout=1)).T.tolist()[0]
    torque = np.array(matlab_engine.eval("torque", nargout=1)).T.tolist()[0]
    GM_velocity = matlab_engine.eval("Gm_velocity", nargout=1)

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
    matlab_engine.eval("clear", nargout=0)
    matlab_engine.eval("clc", nargout=0)

    error = calculate_reward_info_step(sim_data)
    sim_data.update(error)

    return sim_data

def plot_trap_results(error_info):
    try:
        if error_info["Kvi"] % 10 == 0:
            textstr_trap = '\n'.join([
                    f"settling_time_001p : {error_info['settling_time_001p']:.8f}",
                    f"settling_time_01p  : {error_info['settling_time_01p']:.8f}",
                    f"settling_time_1p   : {error_info['settling_time_1p']:.8f}",
                    f"settling_time_3p   : {error_info['settling_time_3p']:.8f}",
                    f"damping_ratio      : {error_info['damping_ratio']:.4f}",
                    f"Ess_trap           : {error_info['Ess_trap']:.8f}", 
                ])
            

            plt.figure(figsize=(10, 6))
            t = np.arange(len(error_info["pos_trap"])) * 0.001
            pos = error_info["pos_trap"]
            pos_cmd = error_info["pos_cmd_trap"]

            # 主圖
            ax = plt.gca()
            ax.plot(t, pos, label='pos')
            ax.plot(t, pos_cmd, label='pos_cmd')
            ax.plot(t[-1500], pos_cmd[-1500], label='Add T_Load', marker='o', markersize=3, color='r')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Position (rad)")
            ax.set_title(f"Position: Kpp={Kpp}, Kvp={Kvp}, Kvi={Kvi}")
            ax.legend(loc="best")

            # 文字方塊
            ax.text(1.02, 0.45, "Sim:\n" + textstr_trap, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.7))

            # ===== 放大區塊 =====
            # 放大時間與位置範圍（可依你畫框調整）
            x1, x2 = t[-1]-1.7, t[-1]+0.1
            y1 = max(pos_cmd)-0.05 if max(pos_cmd) < max(pos) else max(pos)-0.05
            y2 = max(pos_cmd)+0.05 if max(pos_cmd) > max(pos) else max(pos)+0.05

            # 建立小圖
            axins = inset_axes(ax, width="50%", height="50%", loc="center")
            axins.plot(t, pos, label='pos')
            axins.plot(t, pos_cmd, label='pos_cmd')
            axins.plot(t[-1500], pos_cmd[-1500], label='Add T_Load', marker='o', markersize=3, color='r')
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xlabel("Time (s)", fontsize=8)   # 或者省略文字只留下刻度
            axins.set_ylabel("Position (rad)", fontsize=8)
            axins.tick_params(axis='both', which='both', labelsize=7)
            axins.grid(True)

            # 畫框線連接主圖與小圖
            mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="blue", linewidth=1.2)

            plt.tight_layout()

            save_name_trap = "./figure/performance/trap/"
            os.makedirs(save_name_trap, exist_ok=True)
            plt.savefig(save_name_trap+f"Kpp_{Kpp}_Kvp_{Kvp}_Kvi_{Kvi}_pos.png", dpi=150, bbox_inches='tight')
            plt.savefig(save_name_trap+f"Kpp_{Kpp}_Kvp_{Kvp}_Kvi_{Kvi}_pos.svg", dpi=150, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(t, error_info["torque_trap"])
            plt.xlabel("Time (s)")
            plt.ylabel("Torque (Nm)")
            plt.title(f"Torque: Kpp={Kpp}, Kvp={Kvp}, Kvi={Kvi}")
            plt.tight_layout()
            plt.savefig(save_name_trap+f"Kpp_{Kpp}_Kvp_{Kvp}_Kvi_{Kvi}_torque.png", dpi=150, bbox_inches='tight')
            plt.savefig(save_name_trap+f"Kpp_{Kpp}_Kvp_{Kvp}_Kvi_{Kvi}_torque.svg", dpi=150, bbox_inches='tight')
            plt.close()

    except Exception as e:
        print(f"Error occurred for **trap** Kvi={Kvi}: {e}")

def plot_step_results(error_info):
    try:
        if error_info["Kvi"] % 10 == 0:
            textstr_step = '\n'.join([
                    f"overshoot          : {error_info['overshoot']:.8f}",
                    f"overshoot_load     : {error_info['overshoot_load']:.8f}",
                    f"settling_time      : {error_info['settling_time']:.8f}",
                    f"settling_time_load : {error_info['settling_time_load']:.8f}",
                    f"GM_velocity        : {error_info['GM_velocity']:.8f}",
                    f"Ess_step           : {error_info['Ess_step']:.8f}",
                ])

            plt.figure(figsize=(10, 6))
            t = np.arange(len(error_info["pos_step"])) * 0.001
            pos = error_info["pos_step"]
            pos_cmd = error_info["pos_cmd_step"]

            # 主圖
            ax = plt.gca()
            ax.plot(t, pos, label='pos')
            ax.plot(t, pos_cmd, label='pos_cmd')
            ax.plot(t[-1500], pos_cmd[-1500], label='Add T_Load', marker='o', markersize=3, color='r')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Position (rad)")
            ax.set_title(f"Position: Kpp={Kpp}, Kvp={Kvp}, Kvi={Kvi}")
            ax.legend(loc="best")

            # 文字方塊
            ax.text(1.02, 0.45, "Sim:\n" + textstr_step, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.7))

            # ===== 放大區塊 =====
            # 放大時間與位置範圍（可依你畫框調整）
            x1, x2 = t[-1]-1.7, t[-1]+0.1
            y1 = max(pos_cmd)-0.05 if max(pos_cmd) < max(pos) else max(pos)-0.05
            y2 = max(pos_cmd)+0.05 if max(pos_cmd) > max(pos) else max(pos)+0.05

            # 建立小圖
            axins = inset_axes(ax, width="50%", height="50%", loc="center")
            axins.plot(t, pos, label='pos')
            axins.plot(t, pos_cmd, label='pos_cmd')
            axins.plot(t[-1500], pos_cmd[-1500], label='Add T_Load', marker='o', markersize=3, color='r')
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xlabel("Time (s)", fontsize=8)   # 或者省略文字只留下刻度
            axins.set_ylabel("Position (rad)", fontsize=8)
            axins.tick_params(axis='both', which='both', labelsize=7)
            axins.grid(True)

            # 畫框線連接主圖與小圖
            mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="blue", linewidth=1.2)

            plt.tight_layout()

            save_name_step = "./figure/performance/step/"
            os.makedirs(save_name_step, exist_ok=True)
            plt.savefig(save_name_step+f"Kpp_{Kpp}_Kvp_{Kvp}_Kvi_{Kvi}_pos.png", dpi=150, bbox_inches='tight')
            plt.savefig(save_name_step+f"Kpp_{Kpp}_Kvp_{Kvp}_Kvi_{Kvi}_pos.svg", dpi=150, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(t, error_info["torque_step"])
            plt.xlabel("Time (s)")
            plt.ylabel("Torque (Nm)")
            plt.title(f"Torque: Kpp={Kpp}, Kvp={Kvp}, Kvi={Kvi}")
            plt.tight_layout()
            plt.savefig(save_name_step+f"Kpp_{Kpp}_Kvp_{Kvp}_Kvi_{Kvi}_torque.png", dpi=150, bbox_inches='tight')
            plt.savefig(save_name_step+f"Kpp_{Kpp}_Kvp_{Kvp}_Kvi_{Kvi}_torque.svg", dpi=150, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"Error occurred for **step** Kvi={Kvi}: {e}")

def plot_performance(df):
    performance = ["Emax_before", "Eavg_before", "Emax_after", "Eavg_after", "Emax_after_more", "Eavg_after_more", \
                   "settling_time_001p", "settling_time_01p", "settling_time_1p", "settling_time_3p", "damping_ratio", 
                   "Ess_trap", "Ess_step", "overshoot", "settling_time", "max_torque_flag", "GM_velocity", "overshoot_load", "settling_time_load"]

    for key in performance:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(df["Kvi"], df[key], marker='o')
            plt.xlabel("Kvi")
            plt.ylabel(key)
            plt.title(f"Performance: {key} vs Kvi")
            plt.grid()
            plt.tight_layout()
            save_name = "./figure/comparison/"
            os.makedirs(save_name, exist_ok=True)
            plt.savefig(save_name + f"{key}_vs_Kvi.png", dpi=150, bbox_inches='tight')
            plt.savefig(save_name + f"{key}_vs_Kvi.svg", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error occurred while plotting {key}: {e}")

if __name__ == "__main__":

    matlab_engine = matlab.engine.start_matlab()
    matlab_engine.eval("warning('off', 'all');", nargout=0)
    print("matlab engine activate done")

    sys_num = 1
    Kpp = 250.0
    Kvp = 590.0
    # Kvis = np.linspace(0.0, 100.0, 101)
    T_load = 0.5
    # T_load = 1.0

    # sys_num = 3
    # Kpp = 200.0
    # Kvp = 950.0
    # # Kvis = np.linspace(0.0, 100.0, 101)
    # T_load = 0.5
    # T_load = 1.0

    # sys_num = 8
    # Kpp = 30.0
    # Kvp = 90.0
    # Kvis = np.linspace(0.0, 100.0, 101)
    # T_load = 0.5
    # T_load = 1.0

    # sys_num = 9
    # Kpp = 50.0
    # Kvp = 160.0
    # Kvis = np.linspace(0.0, 100.0, 101)
    # Kvis = [0.0, 10.0, 100.0, 200.0, 500.0, 900.0]
    # T_load = 0.5
    # T_load = 1.0

    Kvis = np.linspace(0.0, Kvp / 3, 101)

    all_dicts = []
    for Kvi in Kvis:
        print("Kvi:", Kvi)
        error_info = {}
        error_info.update({
            "Kpp" : Kpp,
            "Kvp" : Kvp,
            "Kvi" : Kvi,
        })

        error_info.update(simulation_trap(Kpp, Kvp, Kvi))
        plot_trap_results(error_info)

        error_info.update(simulation_step(Kpp, Kvp, Kvi))
        plot_step_results(error_info)

        all_dicts.append(error_info)

    df = pd.DataFrame(all_dicts)

    plot_performance(df)

    df.to_csv("./figure/results.csv", index=False)

