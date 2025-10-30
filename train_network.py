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

# 動的な学習率を取得する関数
def get_dynamic_learning_rate(cycle):
    """
    サイクル数に応じて学習率を動的に変更
    cycle 0-9: 0.001 (初期学習)
    cycle 10-19: 0.0005 (中期学習)
    cycle 20-29: 0.0002 (後期学習)
    cycle 30+: 0.0001 (微調整)
    """
    if cycle < 10:
        return 0.001
    elif cycle < 20:
        return 0.0005
    elif cycle < 30:
        return 0.0002
    else:
        return 0.0001

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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f'>> Learning Rate: {learning_rate}')

    # 学習率スケジューラ
    def lr_lambda(epoch):
        if epoch >= 80:
            return 0.25
        elif epoch >= 50:
            return 0.5
        else:
            return 1.0
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

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
        print(f'Epoch {epoch + 1}/{RN_EPOCHS}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

    # 最新プレイヤーのモデルの保存
    torch.save(model.state_dict(), './model/latest.pth')
    print('Model saved to ./model/latest.pth')

    # モデルの破棄
    del model

# 動作確認
if __name__ == '__main__':
    train_network()
