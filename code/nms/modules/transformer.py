import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path


class RelMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        """
        Multi-headed attention with relative positional encoding and
        memory mechanism.

        Args:
            d_model (int): Input dimension.
            n_head (int): Number of heads.
        """
        super(RelMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_model // n_head
        self.atten_scale = 1 / math.sqrt(self.d_model)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

    def skew(self, qpos, mem_len):
        padding_shape = list(qpos.shape)
        padding_shape[-1] = 1
        padding = torch.full(padding_shape, -1e10, device=qpos.device)

        relpos = torch.cat([qpos, padding], dim=-1)
        relpos = relpos.flatten(start_dim=-2)
        relpos = relpos[..., :qpos.shape[-1] * qpos.shape[-2]]
        relpos = relpos.reshape(*qpos.shape)

        q_len = qpos.shape[-2]
        k_len = int((qpos.shape[-1] + 1) / 2)

        assert q_len + mem_len == k_len

        zero_pos_idx = k_len - 1
        start_idx = zero_pos_idx - mem_len
        end_idx = start_idx + k_len

        relpos = relpos[..., start_idx:end_idx]
        return relpos

    def forward(self, hidden, pos_emb, mask=None, memory=None,
                extra_atten_score=None):
        """
        Args:
            hidden (Tensor): Input embedding or hidden state of previous layer.
                Shape: (batch, seq, dim)
            pos_emb (Tensor): Relative positional embedding lookup table.
                Shape: (batch, (seq+mem_len)*2-1, d_head)
                pos_emb[:, seq+mem_len]

            mask (BoolTensor, optional): Attention mask.
                Set item value to True if you DO NOT want keep certain
                attention score, otherwise False. Defaults to None.
                Shape: (seq, seq+mem_len).
            memory (Tensor): Memory tensor of previous layer.
                Shape: (batch, mem_len, dim)
        """

        if memory is None:
            combined = hidden
            mem_len = 0
        else:
            combined = torch.cat([memory, hidden], dim=1)
            mem_len = memory.shape[1]

        # shape: (batch, q/k/v_len, dim)
        q = self.w_q(hidden)
        k = self.w_k(combined)
        v = self.w_v(combined)

        # reshape to (batch, q/k/v_len, n_head, d_head)
        q = q.reshape(q.shape[0], q.shape[1], self.n_head, self.d_head)
        k = k.reshape(k.shape[0], k.shape[1], self.n_head, self.d_head)
        v = v.reshape(v.shape[0], v.shape[1], self.n_head, self.d_head)

        # transpose to (batch, n_head, q/k/v_len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # add n_head dimension for relative positional embedding lookup table
        # (batch, n_head, k/v_len*2-1, d_head)
        pos_emb = pos_emb[:, None]

        # (batch, n_head, q_len, k_len)
        atten_score = torch.matmul(q, k.transpose(-1, -2))

        qpos = torch.matmul(q, pos_emb.transpose(-1, -2))
        # DEBUG
        # ones = torch.zeros(q.shape)
        # ones[:, :, :, 0] = 1.0
        # qpos = torch.matmul(ones, pos_emb.transpose(-1, -2))
        atten_score = atten_score + self.skew(qpos, mem_len)
        atten_score = atten_score * self.atten_scale

        if extra_atten_score is not None:
            atten_score = atten_score + extra_atten_score

        if mask is not None:
            # apply attention mask
            atten_score = atten_score.masked_fill(mask, -1e10)
        atten_score = self.softmax(atten_score)

        # (batch, n_head, q_len, d_head)
        atten_vec = torch.matmul(atten_score, v)
        # (batch, q_len, n_head*d_head)
        atten_vec = atten_vec.transpose(1, 2).flatten(start_dim=-2)

        # linear projection
        out = self.w_o(atten_vec)
        return out, atten_score

class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.w_1(x)
        x = self.relu(x)
        x = self.w_2(x)
        return x



class RelEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.attention = RelMultiHeadAttention(d_model, n_head)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.ffn = PositionWiseFFN(d_model, d_ff)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, rel_pos_emb, padding_mask, memory=None, extra_atten_score=None):
        # multi-head attention
        residual = x
        x, attention_score = self.attention(x, rel_pos_emb, mask=padding_mask, memory=memory, extra_atten_score=extra_atten_score)
        x = self.dropout(x) + residual
        x = self.layer_norm1(x)

        # ffn
        residual = x
        x = self.ffn(x)
        x = self.dropout(x) + residual
        x = self.layer_norm2(x)

        return x, attention_score

class Transformer(nn.Module):
    def __init__(self, config, stats_path):
        super().__init__()

        # load statistics
        self.register_buffer('input_mean', torch.zeros(config.input_dim))
        self.register_buffer('input_std', torch.zeros(config.input_dim))
        self.register_buffer('gt_mean', torch.zeros(config.out_dim))
        self.register_buffer('gt_std', torch.zeros(config.out_dim))
        stats_path = Path(stats_path)
        if stats_path.is_file():
            stats = torch.load(stats_path)
            self.input_mean = torch.concat([stats["target_mean"], stats["guide_mean"]], dim=-1)
            self.input_std = torch.concat([stats["target_std"], stats["guide_std"]], dim=-1)
            self.gt_mean = stats["gt_mean"]
            self.gt_std = stats["gt_std"]

        # init network layers
        self.in_linear = nn.Linear(config.input_dim, config.hidden_dim)

        d_head = config.hidden_dim // config.n_head
        self.rel_pos_layer = nn.Sequential(
            nn.Linear(1, d_head),
            nn.PReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_head, d_head),
            nn.Dropout(config.dropout)
        )

        self.dropout = nn.Dropout(p=config.dropout)
        self.n_head = config.n_head
        self.max_len = config.max_len

        self.encoder_layers = nn.ModuleList()
        for i in range(config.n_layers):
            self.encoder_layers.append(RelEncoderLayer(d_model=config.hidden_dim, n_head=config.n_head, d_ff=config.hidden_dim, dropout=config.dropout))

        self.out_linear = nn.Linear(config.hidden_dim, config.out_dim)

    def forward(self, x, padding_mask=None):
        x = (x - self.input_mean) / self.input_std

        # relative positional embedding
        window_len = x.shape[-2]
        rel_pos_idx = self.get_rel_pos_idx(window_len).to(x.device)
        rel_pos_emb = self.rel_pos_layer(rel_pos_idx)

        x = self.in_linear(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x, attention_score = layer(x, rel_pos_emb, padding_mask)

        out = self.out_linear(x)
        out = out * self.gt_std + self.gt_mean
        return out # (N, d_1, C)

    def get_rel_pos_idx(self, window_len):
        pos_idx = torch.arange(-window_len + 1, window_len, dtype=torch.float32) # (1, seq, 1)
        pos_idx = pos_idx[None, :, None]
        return pos_idx
