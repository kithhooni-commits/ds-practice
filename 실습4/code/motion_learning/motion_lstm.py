import torch
import torch.nn as nn
import torch.nn.functional as F

MOTION_CLASSES = {
    0: "IDLE",
    1: "JAB_STRAIGHT",
    2: "LEFT_HOOK",
    3: "RIGHT_UPPERCUT",
    4: "TWO_HAND_GUARD",
    5: "ENERGY_WAVE"
}

class MotionBiLSTM(nn.Module):
    """
    복싱 및 격투 동작 시계열 인식을 위한 양방향 LSTM 신경망 모델
    입력: (Batch, Seq_Len=30, Input_Dim=63)
    출력: (Batch, Num_Classes=6)
    """
    def __init__(self, input_dim=63, hidden_dim=128, num_layers=2, num_classes=6, dropout=0.3):
        super(MotionBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 입력 정규화 레이어
        self.layer_norm = nn.LayerNorm(input_dim)

        # 양방향 LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 어텐션 가중치 레이어 (중요한 타격 프레임에 가중치 부여)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # 분류 헤드
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Dim)
        x = self.layer_norm(x)
        lstm_out, _ = self.lstm(x)  # lstm_out: (Batch, Seq_Len, hidden_dim * 2)

        # 어텐션 계산
        attn_weights = F.softmax(self.attention(lstm_out), dim=1) # (Batch, Seq_Len, 1)
        context_vector = torch.sum(lstm_out * attn_weights, dim=1) # (Batch, hidden_dim * 2)

        # 최종 클래스 로짓
        out = self.fc(context_vector)
        return out
