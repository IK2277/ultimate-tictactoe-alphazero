# ====================
# パラメータ更新部
# ====================

# パッケージのインポート
from dual_network import DN_INPUT_SHAPE, DualNetwork, device
from pathlib import Path
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# パラメータの準備
RN_EPOCHS = 100 # 学習回数
BATCH_SIZE = 128
NUM_WORKERS = 0  # Windowsではマルチプロセッシングの問題を避けるため0に設定

# 動的な学習率を取得する関数（AlphaZero準拠）
def get_dynamic_learning_rate(cycle):
    """
    サイクル数に応じて学習率を動的に変更
    AlphaZero系の知見を参考に、高めの初期学習率から段階的に減衰
    
    参考:
    - AlphaGo Zero: 1e-2 → 1e-3 → 1e-4 (40万, 60万ステップ)
    - AlphaZero: 2e-2 → 2e-3 → 2e-4 (30万, 50万ステップ)
    - OLIVAW (オセロ): 5e-3 → 1e-3 → 1e-4 (第4, 11世代)
    - KataGo: 6e-5/sample (バッチ256で約1.5e-2/batch)
    
    Ultimate Tic-Tac-Toe用の調整:
    - cycle 0-9: 0.005 (初期学習、OLIVAW準拠)
    - cycle 10-19: 0.001 (中期学習、安定化)
    - cycle 20-29: 0.0002 (後期学習、微調整)
    - cycle 30+: 0.0001 (収束期、精密化)
    """
    if cycle < 10:
        return 0.005  # 5倍に増加（従来0.001 → 0.005）
    elif cycle < 20:
        return 0.001  # 従来0.0005 → 0.001（2倍）
    elif cycle < 30:
        return 0.0002  # 維持
    else:
        return 0.0001  # 維持

# 学習データの読み込み
def load_data():
    history_path = sorted(Path('./data').glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)

# カスタムデータセット
class HistoryDataset(Dataset):
    def __init__(self, xs, y_policies, y_values):
        # (N, H, W, C) -> (N, C, H, W) に変換
        self.xs = np.transpose(xs, (0, 3, 1, 2)).astype(np.float32)
        self.y_policies = y_policies.astype(np.float32)
        self.y_values = y_values.astype(np.float32).reshape(-1, 1)
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.y_policies[idx], self.y_values[idx]

# デュアルネットワークの学習
def train_network(learning_rate=None):
    # 学習率の決定（指定がなければデフォルト値を使用）
    if learning_rate is None:
        learning_rate = 0.001
    
    # 学習データの読み込み
    history = load_data()
    xs, y_policies, y_values = zip(*history)

    # 学習のための入力データのシェイプの変換
    xs = np.array(xs)
    y_policies = np.array(y_policies)
    y_values = np.array(y_values)

    # データセットとデータローダーの作成
    dataset = HistoryDataset(xs, y_policies, y_values)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False  # GPU転送を高速化
    )

    # ベストプレイヤーのモデルの読み込み
    model = DualNetwork().to(device)
    model.load_state_dict(torch.load('./model/best.pth', map_location=device, weights_only=True))

    # 損失関数とオプティマイザの設定
    # target_policies は確率分布(one-hot的)なので、KLDivLossまたはクロスエントロピーを使用
    # PyTorchのCrossEntropyLossはクラスインデックスを期待するため、
    # 確率分布同士の比較には手動でクロスエントロピーを計算
    def policy_loss_fn(pred, target):
        # pred: モデルの出力（softmax済み）
        # target: 教師データの確率分布
        return -torch.sum(target * torch.log(pred + 1e-8)) / pred.size(0)
    
    criterion_value = nn.MSELoss()
    
    # AlphaZero準拠: SGD + Momentum 0.9 + L2正則化
    # 参考: AlphaGo Zero, AlphaZero, Minigoなど全てSGD + Momentum 0.9を使用
    # L2正則化（weight_decay=1e-4）は損失関数に組み込まれていた
    optimizer = optim.SGD(
        model.parameters(), 
        lr=learning_rate, 
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False  # AlphaZeroは標準SGD
    )
    
    print(f'>> Optimizer: SGD (Momentum 0.9, AlphaZero-style)', flush=True)
    print(f'>> Learning Rate: {learning_rate}, Weight Decay: 1e-4', flush=True)

    # 学習率スケジューラ（AlphaZero準拠: より緩やかな減衰）
    # AlphaZeroでは数十万ステップで減衰
    # 100エポック × 約230バッチ = 約23,000ステップ
    # → エポック70, 90で減衰（AlphaZeroの比率を維持）
    def lr_lambda(epoch):
        if epoch >= 90:
            return 0.1    # 最終段階: 1/10に減衰
        elif epoch >= 70:
            return 0.2    # 後期: 1/5に減衰
        else:
            return 1.0    # 初期・中期: フル学習率を維持
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    print(f'>> LR Schedule: Epoch 70 (×0.2), Epoch 90 (×0.1)', flush=True)

    # 学習の実行
    model.train()
    for epoch in range(RN_EPOCHS):
        total_loss = 0
        for batch_idx, (inputs, target_policies, target_values) in enumerate(dataloader):
            inputs = inputs.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)

            # 勾配をゼロにリセット
            optimizer.zero_grad()

            # 順伝播
            pred_policies, pred_values = model(inputs)

            # 損失計算
            loss_policy = policy_loss_fn(pred_policies, target_policies)
            loss_value = criterion_value(pred_values, target_values)
            loss = loss_policy + loss_value

            # 逆伝播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        # エポックごとに学習率を更新
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{RN_EPOCHS}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}', flush=True)

    # 最新プレイヤーのモデルの保存
    torch.save(model.state_dict(), './model/latest.pth')
    print('Model saved to ./model/latest.pth', flush=True)

    # モデルの破棄
    del model

# 動作確認
if __name__ == '__main__':
    train_network()
