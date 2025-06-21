# utils/RL_network.py
import torch
import torch.nn as nn
import numpy as np
import random
import pickle
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class SequenceReplayBuffer:
    """
    支援『整段序列』儲存的 Replay Buffer
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, seq):
        """
        seq: List of (state, action, reward, next_state, done)
             表示一段序列 (episode) 的所有 time steps
        """
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(seq)
    
    def sample(self, batch_size):
        """
        一次抽出多條序列 (batch_size 條)，並組裝成 (batch_size, seq_len, dim)
        """
        sampled_seqs = random.sample(self.buffer, batch_size)
        
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        
        for seq in sampled_seqs:
            # seq: [(s, a, r, s_next, done), (s, a, r, s_next, done), ...]
            s, a, r, s2, d = zip(*seq)  # 解壓成 tuple of list
            state_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_state_batch.append(s2)
            done_batch.append(d)
        
        # 轉成 PyTorch tensor，形狀 [batch_size, seq_len, ...]
        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(-1).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(-1).to(device)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)
    
    def save(self, filepath):
        dir_path = os.path.dirname(filepath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(filepath, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.buffer = pickle.load(f)


class RecurrentValueNetwork(nn.Module):
    """
    RNN 版 Critic：輸入 (state, action) 序列，輸出 Q(s,a) 序列
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(RecurrentValueNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        
        # LSTM 的輸入維度 = state_dim + action_dim
        self.lstm = nn.LSTM(input_size=state_dim + action_dim,
                            hidden_size=hidden_dim,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, state_seq, action_seq, hidden):
        # state_seq, action_seq shape: (batch_size, seq_len, feat_dim)
        sa_seq = torch.cat([state_seq, action_seq], dim=-1)  # 在特徵維度拼接
        out, (hn, cn) = self.lstm(sa_seq, hidden)  # out: (batch_size, seq_len, hidden_dim)
        q_seq = self.fc(out)                       # (batch_size, seq_len, 1)
        return q_seq, (hn, cn)
    
    def init_hidden(self, batch_size):
        # LSTM hidden state shape: (num_layers=1, batch_size, hidden_dim)
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


class RecurrentPolicyNetwork(nn.Module):
    """
    RNN 版 Actor：輸入 state 序列，輸出 action 序列
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(RecurrentPolicyNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_size=state_dim,
                            hidden_size=hidden_dim,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
        self.act_out = nn.Tanh()

    def forward(self, state_seq, hidden):
        # state_seq shape: (batch_size, seq_len, state_dim)
        out, (hn, cn) = self.lstm(state_seq, hidden)
        out = self.fc(out)         # (batch_size, seq_len, action_dim)
        action_seq = self.act_out(out)
        return action_seq, (hn, cn)
    
    def init_hidden(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
    
    def select_action(self, state, hidden):
        """
        用於與環境互動時（線上模式）：單一步 state (shape: (state_dim,))
        """
        # 先 unsqueeze(0) -> (1, state_dim)，再 unsqueeze(0) -> (1, 1, state_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            action_seq, hidden_next = self.forward(state_t, hidden)
        action = action_seq.squeeze(0).squeeze(0).cpu().numpy()
        return action, hidden_next
