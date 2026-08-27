import numpy as np

def generate_motion_sequence(label, seq_len=30):
    """
    라벨에 따른 30프레임 시계열 3D 관절 궤적(Trajectory) 생성
    출력 shape: (seq_len, 63)
    """
    seq = np.zeros((seq_len, 21, 3), dtype=np.float32)
    t = np.linspace(0, 1, seq_len)

    # 기본 손 관절 구조
    base_hand = np.zeros((21, 3), dtype=np.float32)
    base_hand[0] = [0, 0, 0] # 손목
    base_hand[5] = [0.3, 0.8, 0]
    base_hand[9] = [0, 0.9, 0]
    base_hand[13] = [-0.25, 0.85, 0]
    base_hand[17] = [-0.5, 0.75, 0]
    base_hand[8] = [0.3, 1.4, 0] # 검지 끝

    for i in range(seq_len):
        cur_t = t[i]
        pts = base_hand.copy()

        if label == 0:  # IDLE (바운스 스텝)
            offset_y = np.sin(cur_t * 2 * np.pi) * 0.1
            offset_x = np.cos(cur_t * 2 * np.pi) * 0.05
            pts[:, 0] += offset_x
            pts[:, 1] += offset_y

        elif label == 1:  # JAB_STRAIGHT (직선 가속 펀치)
            # 0~0.5초 동안 z축(앞)으로 급격히 가속, 0.5~1초 동안 회수
            strike_z = np.sin(cur_t * np.pi) * 2.5
            strike_x = cur_t * 0.4
            pts[:, 2] -= strike_z
            pts[:, 0] += strike_x

        elif label == 2:  # LEFT_HOOK (좌측 회전 훅)
            # 좌에서 우로 반원 궤적
            angle = cur_t * np.pi
            pts[:, 0] += -np.cos(angle) * 1.8
            pts[:, 2] -= np.sin(angle) * 1.5

        elif label == 3:  # RIGHT_UPPERCUT (하단에서 상단으로 솟구침)
            # 아래로 내려갔다가 위로 강하게 솟구침
            lift_y = np.sin(cur_t * np.pi) * 2.2
            strike_z = np.sin(cur_t * np.pi) * 1.2
            pts[:, 1] += lift_y - 0.5
            pts[:, 2] -= strike_z

        elif label == 4:  # TWO_HAND_GUARD (얼굴 앞 방어 가드)
            # 손목과 관절이 얼굴 중심(0, 0.5, -0.3)으로 밀집
            pts[:, 0] *= 0.6
            pts[:, 1] += 0.4 + np.sin(cur_t * 4 * np.pi) * 0.03
            pts[:, 2] -= 0.3

        elif label == 5:  # ENERGY_WAVE (장풍 충전 후 방출)
            if cur_t < 0.4: # 충전 단계 (진동)
                pts += np.random.normal(0, 0.05, pts.shape)
            else: # 전방 폭발 방출
                burst_t = (cur_t - 0.4) / 0.6
                pts[:, 2] -= burst_t * 3.5
                pts[:, 0] *= 1.5

        # 물리적 센서 노이즈 추가
        pts += np.random.normal(0, 0.02, pts.shape)
        seq[i] = pts

    # (seq_len, 21*3=63) 으로 플래튼
    return seq.reshape(seq_len, -1)


def generate_boxing_dataset(num_samples_per_class=300, seq_len=30):
    np.random.seed(42)
    X = []
    y = []

    for label in range(6):
        for _ in range(num_samples_per_class):
            seq = generate_motion_sequence(label, seq_len)
            X.append(seq)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y
