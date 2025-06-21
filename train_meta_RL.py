import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime
from tqdm import tqdm
import os
import keyboard
import psutil

from utils.meta_RL_network_v2 import (
    device,
    SequenceReplayBuffer,
    RecurrentValueNetwork, RecurrentPolicyNetwork,   # ← LSTM
    GRUCritic, GRUPolicy,                            # ← GRU
    TransformerCritic, TransformerPolicy             # ← Transformer
)

from utils.env import Environment
from utils.save_data import *
from utils.initial_Kpp_count_v2 import last_param_iter

from PIL import Image
import torchvision.transforms as T

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir, "..", "..", "system_parameters", "image_data")
image_path = os.path.abspath(image_path)

def load_spectrum_image(sys_num, root_folder='image_data', size=(128, 128)):
    """
    根據 sys_num 載入對應的 magnitude 與 phase 圖，合併成 2xHxW 的 tensor。
    """
    sys_num = int(sys_num)
    mag_path = os.path.join(image_path, f"{sys_num}/sys{sys_num}_mag.jpg")
    phase_path = os.path.join(image_path, f"{sys_num}/sys{sys_num}_phase.jpg")

    # 轉換為灰階並resize
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor()  # 轉為 [C, H, W]，灰階為 [1, H, W]
    ])

    mag_img = transform(Image.open(mag_path).convert("L"))  # [1, H, W]
    phase_img = transform(Image.open(phase_path).convert("L"))  # [1, H, W]

    spectrum_tensor = torch.cat([mag_img, phase_img], dim=0)  # [2, H, W]
    return spectrum_tensor

def load_spectrum_batch(sys_num_batch, root_folder='image_data', size=(128, 128)):
    """
    根據 sys_num_batch 載入每個對應的 spectrum tensor，組成 [B, 2, 128, 128]
    """
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor()
    ])
    spectrum_batch = []
    for i in range(len(sys_num_batch)):
        sys_num = int(sys_num_batch[i, 0, 0].item())  # 取出數值
        mag_path = os.path.join(image_path, f"{sys_num}/sys{sys_num}_mag.jpg")
        phase_path = os.path.join(image_path, f"{sys_num}/sys{sys_num}_phase.jpg")

        mag_img = transform(Image.open(mag_path).convert("L"))  # [1, H, W]
        phase_img = transform(Image.open(phase_path).convert("L"))  # [1, H, W]

        spectrum_tensor = torch.cat([mag_img, phase_img], dim=0)  # [2, H, W]
        spectrum_batch.append(spectrum_tensor)

    spectrum_batch_tensor = torch.stack(spectrum_batch, dim=0)  # [B, 2, H, W]
    return spectrum_batch_tensor

def train(pretain=False):
    # ----------------- 超參數設定 -----------------
    max_episodes = 8000             # 你原本的 max_episodes => 這裡視為「回合數」 
    max_steps = 20                  # 每回合最多步數
    batch_size = 8                  # 訓練批量 (可自行調整)
    replay_buffer_capacity = 2000   # 容納 2000 條序列
    gamma = 0.99
    soft_tau = 1e-4
    hidden_dim = 128
    value_lr = 1e-3
    policy_lr = 1e-3

    # ----------------- 建立資料夾 -----------------
    model_type = "LSTM"  # "LSTM" or "GRU" or "Transformer"
    # model_type = "GRU"  # "LSTM" or "GRU" or "Transformer"
    # model_type = "Transformer"  # "LSTM" or "GRU" or "Transformer"

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_folder = './record/'
    results_folder =  record_folder + model_type + "_" + current_datetime + '/'
    weights_folder = results_folder + 'weights/'
    temp_weights_folder = record_folder + 'temp_weights/'
    os.makedirs(record_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(weights_folder, exist_ok=True)
    os.makedirs(temp_weights_folder, exist_ok=True)

    # ----------------- 環境實例 -----------------
    env = Environment(9, 3, results_folder=results_folder)
    env.training = True
    state_dim = env.state_dim
    action_dim = env.action_dim

    if model_type == "LSTM":
        value_net = RecurrentValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        policy_net = RecurrentPolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        target_value_net = RecurrentValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        target_policy_net = RecurrentPolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

    elif model_type == "GRU":
        value_net = GRUCritic(state_dim, action_dim, hidden_dim).to(device)
        policy_net = GRUPolicy(state_dim, action_dim, hidden_dim).to(device)
        target_value_net = GRUCritic(state_dim, action_dim, hidden_dim).to(device)
        target_policy_net = GRUPolicy(state_dim, action_dim, hidden_dim).to(device)

    elif model_type == "Transformer":
        value_net = TransformerCritic(state_dim, action_dim, hidden_dim).to(device)
        policy_net = TransformerPolicy(state_dim, action_dim, hidden_dim).to(device)
        target_value_net = TransformerCritic(state_dim, action_dim, hidden_dim).to(device)
        target_policy_net = TransformerPolicy(state_dim, action_dim, hidden_dim).to(device)

    # 複製權重到 target
    target_value_net.load_state_dict(value_net.state_dict())
    target_policy_net.load_state_dict(policy_net.state_dict())

    # Optimizer
    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
    value_criterion = nn.SmoothL1Loss()

    # ----------------- (可選) 是否載入舊模型 -----------------
    if pretain:
        value_net = torch.load(temp_weights_folder + "last_value.pth", weights_only=False)
        policy_net = torch.load(temp_weights_folder + "last_policy.pth", weights_only=False)
        target_value_net = torch.load(temp_weights_folder + "last_target_value.pth", weights_only=False)
        target_policy_net = torch.load(temp_weights_folder + "last_target_policy.pth", weights_only=False)
        print("Load models from temp_weights_folder")

    # ----------------- Sequence Replay Buffer -----------------
    replay_buffer = SequenceReplayBuffer(replay_buffer_capacity)
    if pretain:
        try:
            replay_buffer.load(temp_weights_folder + "buffer.pkl")
            print("Load replay buffer from temp_weights_folder")
        except FileNotFoundError:
            print("Start with an empty buffer.")

    # ----------------- 訓練時需紀錄的東西 -----------------
    value_losses = ['not yet']
    policy_losses = ['not yet']
    episodes_idxs = []
    episode_rewards = []
    episode_controller_parrams = []
    episode_reward_list = []
    episode_sys_list = []

    # ----------------- 函式: rnn_ddpg_update -----------------
    def rnn_ddpg_update():
        """ 一次抽樣 batch_size 條序列，對 RNN Actor/Critic 做 BPTT """
        if len(replay_buffer) < batch_size:
            return

        state_seq, action_seq, reward_seq, next_state_seq, done_seq, sys_num_seq = replay_buffer.sample(batch_size)
        image_seq = load_spectrum_batch(sys_num_seq)
        
        batch_size_, seq_len, _ = state_seq.shape

        # 讀取影像資料 


        # 初始化 LSTM hidden
        if model_type in ["LSTM", "GRU"]:
            v_hidden = value_net.init_hidden(batch_size_)
            p_hidden = policy_net.init_hidden(batch_size_)
            tv_hidden = target_value_net.init_hidden(batch_size_)
            tp_hidden = target_policy_net.init_hidden(batch_size_)
        else:
            v_hidden = p_hidden = tv_hidden = tp_hidden = None  # Transformer 不需要 hidden

        # ---- Critic Update ----
        with torch.no_grad():
            if model_type in ["LSTM", "GRU"]:
                next_action_seq, _ = target_policy_net(next_state_seq, image_seq, tp_hidden)
                target_q_seq, _ = target_value_net(next_state_seq, next_action_seq, image_seq, tv_hidden)
            else:
                next_action_seq, _ = target_policy_net(next_state_seq)
                target_q_seq, _ = target_value_net(next_state_seq, next_action_seq)

            td_target = reward_seq + (1.0 - done_seq) * gamma * target_q_seq

        if model_type in ["LSTM", "GRU"]:
            current_q_seq, _ = value_net(state_seq, action_seq, image_seq, v_hidden)
            pred_action_seq, _ = policy_net(state_seq, image_seq, p_hidden)
            pred_q_seq, _ = value_net(state_seq, pred_action_seq, image_seq, v_hidden)
        else:
            current_q_seq, _ = value_net(state_seq, action_seq)
            pred_action_seq, _ = policy_net(state_seq)
            pred_q_seq, _ = value_net(state_seq, pred_action_seq)

        value_loss = value_criterion(current_q_seq, td_target)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # ---- Actor Update ----
        pred_action_seq, _ = policy_net(state_seq, image_seq, p_hidden)
        pred_q_seq, _ = value_net(state_seq, pred_action_seq, image_seq, v_hidden)
        policy_loss = -pred_q_seq.mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # ---- Soft update target networks ----
        for t_param, param in zip(target_value_net.parameters(), value_net.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        for t_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - soft_tau) + param.data * soft_tau)

        value_losses.append(value_loss.item())
        policy_losses.append(policy_loss.item())

    # ----------------- Checkpoint 存檔 -----------------
    def save_checkpoint(episodes):
        torch.save(value_net, temp_weights_folder + "last_value.pth")
        torch.save(policy_net, temp_weights_folder + "last_policy.pth")
        torch.save(target_value_net, temp_weights_folder + "last_target_value.pth")
        torch.save(target_policy_net, temp_weights_folder + "last_target_policy.pth")
        
        torch.save(value_net, weights_folder + "last_value.pth")
        torch.save(policy_net, weights_folder + "last_policy.pth")
        torch.save(target_value_net, weights_folder + "last_target_value.pth")
        torch.save(target_policy_net, weights_folder + "last_target_policy.pth")

        replay_buffer.save(weights_folder + "buffer.pkl")
        replay_buffer.save(temp_weights_folder + "buffer.pkl")

        save_result(episodes_idxs, episode_rewards, value_losses, policy_losses,
                    results_folder, episode_controller_parrams, episode_reward_list)
        with open(results_folder + 'time.txt', 'w') as f:
            f.write(f"Training interrupted at frame {episodes}. Total time: {time.time() - start_time} s")
        np.savetxt(results_folder + 'sys_num.txt', np.array(episode_sys_list))

    # ----------------- 開始訓練 -----------------
    print(device)
    start_time = time.time()

    with tqdm(total=max_episodes, desc=f"{model_type}-DDPG Training", unit="episode") as pbar_out:
        for episodes in range(max_episodes):
            with tqdm(total=max_steps, desc=f"episode {episodes}/{max_episodes}", unit="step", leave=False) as pbar:
                env.change_system_param()
                image_spectrum = load_spectrum_image(env.sys_num).unsqueeze(0).float()

                state = env.reset()

                # 紀錄 episode 交互資料
                seq_data = []
                episode_reward = np.zeros(23)
                
                episodes_idxs.append(episodes)
                
                # 紀錄控制器參數
                controller_parrams = [env.controller_param.copy()]

                # Actor hidden (線上) 初始化
                if model_type in ["LSTM", "GRU"]:
                    actor_hidden = policy_net.init_hidden(batch_size=1)

                done = 0
                for step_idx in range(max_steps):
                    # 取動作
                    if done == 1:
                        break
                    else:
                        if model_type in ["LSTM", "GRU"]:
                            action, actor_hidden = policy_net.select_action(state, image_spectrum, actor_hidden)
                        elif model_type == "Transformer":
                            action, _ = policy_net.select_action(state)


                    # 執行一步
                    old_state, next_state, reward = env.step(state, action)

                    done = 0
                    if env.check_terminal_condition():
                        done = 1

                    seq_data.append((old_state, action, reward, next_state, done, env.sys_num))
                    
                    state = next_state
                    episode_reward += np.array(env.current_reward)
                    controller_parrams.append(env.controller_param.copy())

                    # 更新進度條
                    pbar.set_postfix({
                        "Memory": f"{get_memory_usage():.2f} MB",
                        "Episode Reward": episode_reward[8]})
                    pbar.update(1)

                # 將整個episode存到 Buffer
                replay_buffer.push(seq_data)

                # 做一次更新(或可做多次)
                if len(replay_buffer) >= batch_size:
                    rnn_ddpg_update()

                # 計算該回合的平均 reward
                avg_reward = episode_reward[-1] / (step_idx + 1)
                episode_rewards.append((episode_reward/(step_idx+1)).tolist().copy())
                episode_controller_parrams.append(controller_parrams.copy())
                episode_sys_list.append(env.sys_num)
                episode_reward_list.append(env.reward_list_record.copy())

                # 定期存檔 (每 50 回合)
                if (episodes % 50 == 0) and len(replay_buffer) >= batch_size:
                    torch.save(policy_net, weights_folder + f"{episodes}.pth")
                    save_checkpoint(episodes)
                    last_param_iter(results_folder, plot_flag=False)
            pbar_out.set_postfix({
                "Epoch": f"{episodes}/{max_episodes}",
                "Avg Reward": f"{episode_reward[8]/(step_idx+1):.2f}",
                "Value Loss": f"{value_losses[-1]:.2f}" if isinstance(value_losses[-1], float) else f"{value_losses[-1]}",
                "Policy Loss": f"{policy_losses[-1]:.2f}" if isinstance(policy_losses[-1], float) else f"{policy_losses[-1]}\n"
            })
            pbar_out.update(1)

    # 訓練結束後再存一次
    save_checkpoint(episodes)
    last_param_iter(results_folder)
    print("Training finished, total time = ", time.time() - start_time)

if __name__ == "__main__":
    pretain = True
    pretain = False
    train(pretain=pretain)
