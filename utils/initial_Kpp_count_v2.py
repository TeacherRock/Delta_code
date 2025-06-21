import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def initial_Kvp_count(folder_path):

    data = pd.read_excel(folder_path + "/iterations_params.xlsx", sheet_name='Sheet')

    save_folder = folder_path + "/count"
    os.makedirs(save_folder, exist_ok=True)

    # Extract the first 'Kpp' value for each iteration
    kpp_values = []
    kvp_values = []
    inside_iteration = False

    for _, row in data.iterrows():
        if row['Iteration 1'] == 'Kpp':
            inside_iteration = True  # Start of a new iteration
            continue
        if inside_iteration and pd.notna(row['Iteration 1']):
            try:
                kpp_values.append(float(row['Iteration 1']))
                inside_iteration = False  # End search for this iteration's first Kpp value
            except ValueError:
                pass

    inside_iteration = False
    for _, row in data.iterrows():
        if row['Unnamed: 1'] == 'Kvp':
            inside_iteration = True  # Start of a new iteration
            continue
        if inside_iteration and pd.notna(row['Unnamed: 1']):
            try:
                kvp_values.append(float(row['Unnamed: 1']))
                inside_iteration = False  # End search for this iteration's first Kpp value
            except ValueError:
                pass


    kpp_groups = {
        "(20.0, 40.0)": [],
        "(40.0, 60.0)": [],
        "(60.0, 80.0)": [],
        "(80.0, 100.0)": []
    }

    for idx, kpp in enumerate(kpp_values):
        if 20.0 <= kpp <= 40.0:
            kpp_groups["(20.0, 40.0)"].append(kvp_values[idx])
        elif 40.0 < kpp <= 60.0:
            kpp_groups["(40.0, 60.0)"].append(kvp_values[idx])
        elif 60.0 < kpp <= 80.0:
            kpp_groups["(60.0, 80.0)"].append(kvp_values[idx])
        elif 80.0 < kpp <= 100.0:
            kpp_groups["(80.0, 100.0)"].append(kvp_values[idx])

    # Create and save histograms for each group
    for group, values in kpp_groups.items():
        if values:
            plt.hist(values, bins=10, edgecolor='black')
            plt.title(f'Kpp Value Distribution: {group}')
            plt.xlabel('Kvp Values')
            plt.ylabel('Count')
            plt.tight_layout()
            group_label = group.replace("(", "").replace(")", "").replace(".", "").replace(",", "_").replace(" ", "")
            plt.savefig(f'{save_folder}/Kvp_distr_{group_label}.svg', bbox_inches='tight')
            plt.savefig(f'{save_folder}/Kvp_distr_{group_label}.jpg', bbox_inches='tight')
            plt.clf()

def last_param_iter(folder_path, plot_flag=True):

    data = pd.read_excel(folder_path + "iterations_params.xlsx", sheet_name='Sheet')
    sys_num_list = np.loadtxt(folder_path + "sys_num.txt")

    save_folder = folder_path + "count"
    os.makedirs(save_folder, exist_ok=True)

    kpp_values = []
    kvp_values = []
    temp_kpp_list = []
    temp_kvp_list = []
    inside_iteration = False

    for _, row in data.iterrows():
        if row['Iteration 1'] == 'Kpp':
            # New iteration starts
            if temp_kpp_list:  # Save the last Kpp value of the previous iteration
                kpp_values.append(temp_kpp_list[-1])
            temp_kpp_list = [] 
            inside_iteration = True
            continue
        
        if inside_iteration and pd.notna(row['Iteration 1']):
            try:
                temp_kpp_list.append(float(row['Iteration 1']))  # Add to the temporary list
            except ValueError:
                pass

    if temp_kpp_list:
        kpp_values.append(temp_kpp_list[-1])

    inside_iteration = False
    for _, row in data.iterrows():
        if row['Unnamed: 1'] == 'Kvp':
            if temp_kvp_list:  # Save the last Kvp value of the previous iteration
                kvp_values.append(temp_kvp_list[-1])
            temp_kvp_list = [] 
            inside_iteration = True  # Start of a new iteration
            continue
        if inside_iteration and pd.notna(row['Unnamed: 1']):
            try:
                temp_kvp_list.append(float(row['Unnamed: 1']))
                # inside_iteration = False  # End search for this iteration's first Kpp value
            except ValueError:
                pass

    if temp_kvp_list:
        kvp_values.append(temp_kvp_list[-1])
    
    if plot_flag:
        for i in range(70):
            plt.figure(figsize=(10, 6))
            plt.plot(np.array(kpp_values)[sys_num_list==float(i+1)], linestyle='-', color='b', label='Last Kpp Value')
            plt.title('Last Kpp Value per Iteration')
            plt.xlabel('Iteration')
            plt.ylabel('Last Kpp Value')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{save_folder}/last_Kpp_distr_sys{i+1}.svg', bbox_inches='tight')
            plt.savefig(f'{save_folder}/last_Kpp_distr_sys{i+1}.jpg', bbox_inches='tight')
            plt.clf()

            plt.figure(figsize=(10, 6))
            plt.plot(np.array(kvp_values)[sys_num_list==float(i+1)], linestyle='-', color='b', label='Last Kvp Value')

            plt.title('Last Kvp Value per Iteration')
            plt.xlabel('Iteration')
            plt.ylabel('Last Kvp Value')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{save_folder}/last_Kvp_distr_sys{i+1}.svg', bbox_inches='tight')
            plt.savefig(f'{save_folder}/last_Kvp_distr_sys{i+1}.jpg', bbox_inches='tight')
            plt.clf()
            plt.close('all')
    else:
        for i in range(70):
            np.savetxt(f'{save_folder}/last_Kpp_distr_sys{i+1}.txt', np.array(kpp_values)[sys_num_list==float(i+1)])
            np.savetxt(f'{save_folder}/last_Kvp_distr_sys{i+1}.txt', np.array(kvp_values)[sys_num_list==float(i+1)])

if __name__ == "__main__":
    path = "./record/20250302_154726_劉"
    initial_Kvp_count(path)
    last_kvp_iter(path)