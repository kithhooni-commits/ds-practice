import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from motion_lstm import MotionBiLSTM, MOTION_CLASSES
from synthetic_boxing_data import generate_boxing_dataset

def evaluate_rule_based_baseline(X_test, y_test):
    """
    [베이스라인] 단일 프레임 정적 위치 기반 룰 모델 (시간 궤적을 보지 못함)
    """
    correct = 0
    total = len(y_test)

    for i in range(total):
        # 마지막 프레임의 위치만 보고 단순 휴리스틱 판정
        last_frame = X_test[i, -1].reshape(21, 3)
        z_depth = last_frame[8, 2] # 검지 끝 깊이
        y_height = last_frame[8, 1] # 검지 끝 높이

        if z_depth < -1.5:
            pred = 1 # JAB
        elif y_height > 1.8:
            pred = 3 # UPPERCUT
        elif abs(last_frame[8, 0]) > 1.2:
            pred = 2 # HOOK
        elif z_depth < -0.2 and y_height < 1.0:
            pred = 4 # GUARD
        else:
            pred = 0 # IDLE

        if pred == y_test[i]:
            correct += 1

    return correct / total


def train_motion_model():
    print("=" * 65)
    print("🥊 [GPU Deep Learning] 시계열 Bi-LSTM 복싱 모션 모델 학습 (PyTorch)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 학습 디바이스: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. 데이터셋 생성 (6개 클래스 x 400개 = 2400 시퀀스)
    print("[*] 3D 시계열 궤적 데이터셋 생성 중 (2,400 시퀀스)...")
    X, y = generate_boxing_dataset(num_samples_per_class=400, seq_len=30)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 2. 베이스라인 성능 측정
    rule_acc = evaluate_rule_based_baseline(X_test, y_test)
    print(f"\n[-] [Step 1 정적 2D 키포인트 룰베이스] 정확도: {rule_acc * 100:.2f}%")

    # 3. DataLoader 구성
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 4. 모델 초기화
    model = MotionBiLSTM(input_dim=63, hidden_dim=128, num_layers=2, num_classes=6, dropout=0.25).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    # 5. GPU 학습 루프
    print("\n[*] Bi-LSTM 학습 시작 (15 Epochs)...")
    start_time = time.time()

    for epoch in range(1, 16):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if epoch % 5 == 0 or epoch == 15:
            print(f"    Epoch [{epoch:02d}/15] - Loss: {total_loss / len(train_loader):.4f}")

    train_duration = time.time() - start_time
    print(f"[✓] 학습 완료! 소요 시간: {train_duration:.2f}초")

    # 6. 테스트 평가
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())

    lstm_acc = accuracy_score(all_targets, all_preds)
    print(f"\n[+] [Step 2 시계열 Bi-LSTM (GPU)] 테스트 정확도: {lstm_acc * 100:.2f}%")
    print(f"[★] 정확도 향상: +{(lstm_acc - rule_acc) * 100:.2f}%p 개선! (Show Numbers)")

    print("\n--- Classification Report ---")
    print(classification_report(all_targets, all_preds, target_names=[MOTION_CLASSES[i] for i in range(6)]))

    # 7. 가중치 및 평가 지표 저장
    save_dir = os.path.dirname(__file__)
    model_path = os.path.join(save_dir, "boxing_lstm.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[✓] GPU 학습 가중치 저장 완료: {model_path}")

    metrics = {
        "device": str(device),
        "rule_based_accuracy": float(rule_acc),
        "lstm_accuracy": float(lstm_acc),
        "improvement_pct_points": float((lstm_acc - rule_acc) * 100),
        "training_time_seconds": float(train_duration),
        "test_samples": len(all_targets)
    }
    with open(os.path.join(save_dir, "eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    train_motion_model()
