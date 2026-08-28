"""
tcn_model.py — 실시간 causal TCN (Temporal Convolutional Network)

BiLSTM(motion_lstm.py)과 달리 미래 프레임을 보지 않는다(causal).
각 conv 는 "왼쪽(과거)으로만" padding 한 뒤 오른쪽 잉여분을 잘라내(Chomp) 인과성을 보장하고,
dilation 을 layer마다 2배씩 늘려 적은 layer 수로도 긴 시간창을 커버한다.

입력: (Batch, Seq_Len, Input_Dim=70)  — real_data.py 의 game_7j_temporal_v2
출력: (Batch, Num_Classes=10)
"""
import torch
import torch.nn as nn

from real_data import CLASSES

NUM_CLASSES = len(CLASSES)


class Chomp1d(nn.Module):
    """causal padding 후 오른쪽에 남는 잉여 시점을 잘라낸다."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class CausalMotionTCN(nn.Module):
    def __init__(self, input_dim=17, channels=(32, 32, 32), kernel_size=3,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)

        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Dim)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)          # (Batch, Input_Dim, Seq_Len) — Conv1d 규약
        out = self.tcn(x)              # (Batch, Channels, Seq_Len)
        last = out[:, :, -1]           # causal: 마지막 시점이 지금까지의 전체 과거를 반영
        return self.fc(last)
