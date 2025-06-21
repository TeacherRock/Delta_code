from utils.inference import inference, draw_inference
from utils.initial_Kpp_count_v2 import initial_Kvp_count, last_param_iter

import pandas as pd
import numpy as np
import os

def main_inference(root_folder="./record", inference_folder_name=None):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        if os.path.isdir(folder_path) and folder_name.startswith(inference_folder_name):
            last_param_iter(folder_path + "/")
            print(f"Processing folder: {folder_name}")
            # New Sys 1 ~ 10
            excluded = {}

            current_dir = os.path.dirname(__file__)
            csv_path = os.path.join(current_dir, "..", "..", "system_parameters", "inference_settings.csv")
            csv_path = os.path.abspath(csv_path)

            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                ini_param = [np.array([float(row["ini_Kpp"])]), np.array([float(row["ini_Kvp"])]), np.array([0.0])]
                sys_num = int(row["sys_num"])

                if sys_num not in excluded:
                    save_folder = folder_name + f"/inference/sys_{sys_num}/Kpp_" + str(int(ini_param[0][0])) + "/Kvp_" + str(int(ini_param[1][0]))
                    print("save_folder : ", save_folder)
                    terminal_step_idx, terminal_type = inference(folder_name, ini_param=ini_param, save_folder=save_folder, sys_num=sys_num)
                    draw_inference(save_folder, terminal_step_idx, terminal_type)

def main_testing(root_folder="./record", inference_folder_name=None):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        if os.path.isdir(folder_path) and folder_name.startswith(inference_folder_name):
            last_param_iter(folder_path + "/")
            print(f"Processing folder: {folder_name}")
            excluded = {}

            current_dir = os.path.dirname(__file__)
            csv_path = os.path.join(current_dir, "..", "..", "system_parameters", "test_settings.csv")
            csv_path = os.path.abspath(csv_path)

            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                ini_param = [np.array([float(row["ini_Kpp"])]), np.array([float(row["ini_Kvp"])]), np.array([0.0])]
                sys_num = int(row["sys_num"])

                if sys_num not in excluded:
                    save_folder = folder_name + f"/inference/sys_{sys_num}/Kpp_" + str(int(ini_param[0][0])) + "/Kvp_" + str(int(ini_param[1][0]))
                    print("save_folder : ", save_folder)
                    terminal_step_idx, terminal_type = inference(folder_name, ini_param=ini_param, save_folder=save_folder, sys_num=sys_num)
                    draw_inference(save_folder, terminal_step_idx, terminal_type)


if __name__ == "__main__":
    inference_folder_name = "LSTM_20250608_151201"
    main_inference(inference_folder_name=inference_folder_name)
    # main_testing(inference_folder_name=inference_folder_name)