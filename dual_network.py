# ====================
# デュアルネットワークの作成
# ====================

# パッケージのインポート
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# パラメータの準備
DN_FILTERS  = 128 # 畳み込み層のカーネル数（本家は256）
DN_RESIDUAL_NUM =  16 # 残差ブロックの数（本家は19）
DN_INPUT_SHAPE = (9,9,3) # 入力シェイプ (H, W, C)
DN_OUTPUT_SIZE = 81 # 行動数(配置先(3*3))

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # GPU最適化設定
    torch.backends.cudnn.benchmark = True  # cuDNNの自動チューニングを有効化
    torch.backends.cudnn.deterministic = False  # 再現性より速度を優先
else:
    print("Using CPU")

# 残差ブロックの定義
class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = F.relu(x)
        return x

# デュアルネットワークの定義
class DualNetwork(nn.Module):
    def __init__(self, input_shape=DN_INPUT_SHAPE, filters=DN_FILTERS, 
                 residual_num=DN_RESIDUAL_NUM, output_size=DN_OUTPUT_SIZE):
        super(DualNetwork, self).__init__()
        
        # 入力は (H, W, C) = (9, 9, 3) だが、PyTorchは (C, H, W) = (3, 9, 9) を期待
        in_channels = input_shape[2]
        
        # 初期畳み込み層
        self.conv_input = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(filters)
        
        # 残差ブロック
        self.residual_blocks = nn.ModuleList([ResidualBlock(filters) for _ in range(residual_num)])
        
        # ポリシーヘッド
        self.policy_conv = nn.Conv2d(filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 9 * 9, output_size)
        
        # バリューヘッド
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 9 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # ドロップアウト層（過学習防止）
        self.dropout = nn.Dropout(0.3)
        
        # 重みの初期化
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 入力: (N, C, H, W) = (N, 3, 9, 9)
        
        # 初期畳み込み
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = F.relu(x)
        
        # 残差ブロック
        for block in self.residual_blocks:
            x = block(x)
        
        # ポリシー出力
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        # 非連続メモリ対応で安全にフラット化
        p = torch.flatten(p, 1)
        p = self.dropout(p)  # ドロップアウト適用
        p = self.policy_fc(p)
        p = F.softmax(p, dim=1)
        
        # バリュー出力
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        # 非連続メモリ対応で安全にフラット化
        v = torch.flatten(v, 1)
        v = self.value_fc1(v)
        v = F.relu(v)
        v = self.dropout(v)  # ドロップアウト適用
        v = self.value_fc2(v)
        v = torch.tanh(v)
        
        return p, v

# デュアルネットワークの作成と保存
def dual_network():
    # モデル作成済みの場合は無処理
    if os.path.exists('./model/best.pth'):
        return

    # モデルの作成
    model = DualNetwork().to(device)
    
    # モデルの保存
    os.makedirs('./model/', exist_ok=True)
    torch.save(model.state_dict(), './model/best.pth')
    print("Model saved to './model/best.pth'")

# 動作確認
if __name__ == '__main__':
    dual_network()
