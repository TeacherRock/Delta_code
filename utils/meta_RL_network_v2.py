# ============================
# File: metaRL_network.py
# 整合 LSTM / GRU / Transformer 版本的 Actor-Critic
# ============================
import torch
import torch.nn as nn
import numpy as np
import random
import pickle
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# ----------------------------
# SequenceReplayBuffer
# ----------------------------
class SequenceReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, seq):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(seq)

    def pad_sequence(self, seq, max_len):
        seq = list(seq)
        pad_len = max_len - len(seq)
        if pad_len > 0:
            last = seq[-1]
            seq += [last] * pad_len  # repeat last step to pad
        return seq[:max_len]  # just in case it’s longer

    def sample(self, batch_size):
        sampled_seqs = random.sample(self.buffer, batch_size)
        max_len = max(len(seq) for seq in sampled_seqs)  # or fix to self.max_len

        s, a, r, s2, d, sys_num = [], [], [], [], [], []

        for seq in sampled_seqs:
            ss, aa, rr, ss2, dd, num = zip(*self.pad_sequence(seq, max_len))
            s.append(ss)
            a.append(aa)
            r.append(rr)
            s2.append(ss2)
            d.append(dd)
            sys_num.append(num)

        state_batch = torch.FloatTensor(s).to(device)
        action_batch = torch.FloatTensor(a).to(device)
        reward_batch = torch.FloatTensor(r).unsqueeze(-1).to(device)
        next_state_batch = torch.FloatTensor(s2).to(device)
        done_batch = torch.FloatTensor(d).unsqueeze(-1).to(device)
        sys_num_batch = torch.FloatTensor(sys_num).unsqueeze(-1).to(device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, sys_num_batch


    def __len__(self):
        return len(self.buffer)

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.buffer = pickle.load(f)

# ----------------------------
# LSTM Actor-Critic
# ----------------------------
class SpectralCNN(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),  # (B, 2, 128, 128) -> (B, 16, 64, 64)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # (B, 32, 4, 4)
            nn.Flatten(),  # (B, 512)
            nn.Linear(32 * 4 * 4, out_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.cnn(x)

class RecurrentPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, spectrum_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spectral_encoder = SpectralCNN(out_dim=spectrum_dim)
        self.lstm = nn.LSTM(state_dim + spectrum_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.LayerNorm(int(hidden_dim / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(hidden_dim / 2), action_dim)
        )
        self.act_out = nn.Tanh()

    def forward(self, state_seq, spectrum_img, hidden):
        B, T, _ = state_seq.shape
        spectrum_feat = self.spectral_encoder(spectrum_img.to(device))  # (B, spectrum_dim)
        spectrum_feat_expanded = spectrum_feat.unsqueeze(1).expand(-1, T, -1)
        state_aug = torch.cat([state_seq, spectrum_feat_expanded], dim=-1)
        out, (hn, cn) = self.lstm(state_aug, hidden)
        act_seq = self.act_out(self.fc(out))
        return act_seq, (hn, cn)

    def init_hidden(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(next(self.parameters()).device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(next(self.parameters()).device)
        return (h0, c0)
    
    def select_action(self, state, spectrum_img, hidden):
        state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            action_seq, hidden_next = self.forward(state_t, spectrum_img.to(device), hidden)
        return action_seq.squeeze(0).squeeze(0).cpu().numpy(), hidden_next

class RecurrentValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, spectrum_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spectral_encoder = SpectralCNN(out_dim=spectrum_dim)
        self.lstm = nn.LSTM(state_dim + action_dim + spectrum_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.LayerNorm(int(hidden_dim / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(hidden_dim / 2), 1)
        )

    def forward(self, state_seq, action_seq, spectrum_img, hidden):
        B, T, _ = state_seq.shape
        spectrum_feat = self.spectral_encoder(spectrum_img.to(device))  # (B, spectrum_dim)
        spectrum_feat_expanded = spectrum_feat.unsqueeze(1).expand(-1, T, -1)
        sa_seq = torch.cat([state_seq, action_seq, spectrum_feat_expanded], dim=-1)
        out, (hn, cn) = self.lstm(sa_seq, hidden)
        q_seq = self.fc(out)
        return q_seq, (hn, cn)

    def init_hidden(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(next(self.parameters()).device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(next(self.parameters()).device)
        return (h0, c0)

# ----------------------------
# GRU Actor-Critic
# ----------------------------
class GRUCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(state_dim + action_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.LayerNorm(int(hidden_dim/2)),
            nn.LeakyReLU(),
            nn.Linear(int(hidden_dim/2), 1)
        )

    def forward(self, state_seq, action_seq, hidden):
        x = torch.cat([state_seq, action_seq], dim=-1)
        out, hn = self.gru(x, hidden)
        q_seq = self.fc(out)
        return q_seq, hn

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.gru.hidden_size).to(device)

class GRUPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.LayerNorm(int(hidden_dim/2)),
            nn.LeakyReLU(),
            nn.Linear(int(hidden_dim/2), action_dim)
        )
        self.act = nn.Tanh()

    def forward(self, state_seq, hidden):
        out, hn = self.gru(state_seq, hidden)
        act_seq = self.act(self.fc(out))
        return act_seq, hn

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.gru.hidden_size).to(device)

    def select_action(self, state, hidden):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            action_seq, next_hidden = self.forward(state, hidden)
        return action_seq.squeeze().cpu().numpy(), next_hidden

# ----------------------------
# Transformer Actor-Critic
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, nhead=4, nlayers=2):
        super().__init__()
        self.input_proj = nn.Linear(state_dim + action_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.LayerNorm(int(hidden_dim/2)),
            nn.LeakyReLU(),
            nn.Linear(int(hidden_dim/2), 1)
        )

    def forward(self, state_seq, action_seq, hidden=None):
        x = torch.cat([state_seq, action_seq], dim=-1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        return self.fc_out(x), None

class TransformerPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, nhead=4, nlayers=2):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.LayerNorm(int(hidden_dim/2)),
            nn.LeakyReLU(),
            nn.Linear(int(hidden_dim/2), action_dim)
        )
        self.act = nn.Tanh()

    def forward(self, state_seq, hidden=None):
        x = self.input_proj(state_seq)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        return self.act(self.fc_out(x)), None

    def select_action(self, state, hidden=None):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            action_seq, _ = self.forward(state)
        return action_seq.squeeze().cpu().numpy(), None