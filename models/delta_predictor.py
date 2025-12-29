"""
差分予測モデル（Δf → Δg）

オノマトペ特徴量の差分から、音声latentの差分を予測するMLPモデル。
これがシステムの中核となるモデル。
"""
import torch
import torch.nn as nn
from typing import List, Optional


class ResidualBlock(nn.Module):
    """残差接続付きのMLPブロック"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.dropout(self.block(x)))


class DeltaPredictor(nn.Module):
    """
    オノマトペ差分 → 音声latent差分 の予測モデル

    入力: Δf = f(z2) - f(z1)  (オノマトペ特徴量の差分)
    出力: Δg' = predict(g(l2) - g(l1))  (音声latentの差分予測)

    アーキテクチャ:
    - 入力層: D次元（オノマトペ特徴量次元）
    - 隠れ層: 複数のMLPブロック（残差接続付き）
    - 出力層: latent_dim × latent_length 次元（flatten済み）
    """

    def __init__(
        self,
        input_dim: int,           # オノマトペ特徴量の次元
        latent_dim: int = 64,     # latentのチャンネル数
        latent_length: int = 32,  # latentの時間長
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        """
        Args:
            input_dim: 入力次元（オノマトペ特徴量差分の次元）
            latent_dim: latentのチャンネル次元
            latent_length: latentの時間方向の長さ
            hidden_dims: 隠れ層の次元リスト
            dropout: ドロップアウト率
            use_residual: 残差接続を使用するか
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.latent_length = latent_length
        self.output_dim = latent_dim * latent_length
        self.use_residual = use_residual

        if hidden_dims is None:
            hidden_dims = [1024, 2048, 4096]

        # 入力層
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 隠れ層
        layers = []
        for i in range(len(hidden_dims) - 1):
            if use_residual and hidden_dims[i] == hidden_dims[i + 1]:
                # 次元が同じなら残差ブロック
                layers.append(ResidualBlock(hidden_dims[i], dropout))
            else:
                # 次元が異なるなら通常の線形層
                layers.append(nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LayerNorm(hidden_dims[i + 1]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ))

        self.hidden_layers = nn.Sequential(*layers)

        # 出力層
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], self.output_dim),
        )

        # 出力の正規化用（学習時に更新）
        self.register_buffer('output_mean', torch.zeros(self.output_dim))
        self.register_buffer('output_std', torch.ones(self.output_dim))

        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, delta_f: torch.Tensor) -> torch.Tensor:
        """
        オノマトペ差分からlatent差分を予測

        Args:
            delta_f: shape (batch, input_dim) - オノマトペ特徴量の差分

        Returns:
            delta_g: shape (batch, latent_dim, latent_length) - latent差分の予測
        """
        batch_size = delta_f.shape[0]

        # 順伝播
        x = self.input_proj(delta_f)
        x = self.hidden_layers(x)
        x = self.output_proj(x)

        # reshape: (batch, output_dim) -> (batch, latent_dim, latent_length)
        delta_g = x.view(batch_size, self.latent_dim, self.latent_length)

        return delta_g

    def predict_target_latent(
        self,
        latent1: torch.Tensor,
        delta_f: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        元latentとオノマトペ差分から目標latentを予測

        g(l2)' = g(l1) + α * Δg_pred

        Args:
            latent1: shape (batch, latent_dim, latent_length) - 元latent
            delta_f: shape (batch, input_dim) - オノマトペ差分
            alpha: スケール係数

        Returns:
            latent2_pred: shape (batch, latent_dim, latent_length) - 目標latent予測
        """
        delta_g = self.forward(delta_f)
        return latent1 + alpha * delta_g

    def set_output_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        """出力の正規化パラメータを設定"""
        self.output_mean = mean.view(-1)
        self.output_std = std.view(-1)


class DeltaPredictorLoss(nn.Module):
    """
    差分予測モデルの損失関数

    L = MSE(Δg_pred, Δg_true) + λ * ||Δg_pred||^2
    """

    def __init__(self, norm_weight: float = 0.01):
        """
        Args:
            norm_weight: 正則化項の重み
        """
        super().__init__()
        self.norm_weight = norm_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        delta_g_pred: torch.Tensor,
        delta_g_true: torch.Tensor,
    ) -> dict:
        """
        損失を計算

        Args:
            delta_g_pred: 予測latent差分
            delta_g_true: 正解latent差分

        Returns:
            損失の辞書 {'total': ..., 'mse': ..., 'norm': ...}
        """
        # MSE損失
        mse_loss = self.mse(delta_g_pred, delta_g_true)

        # 正則化項（予測の大きさを抑制）
        norm_loss = torch.mean(delta_g_pred ** 2)

        # 合計損失
        total_loss = mse_loss + self.norm_weight * norm_loss

        return {
            'total': total_loss,
            'mse': mse_loss,
            'norm': norm_loss,
        }


if __name__ == "__main__":
    # テスト
    batch_size = 4
    input_dim = 1280  # オノマトペ特徴量次元 (embedding_dim * max_phoneme_length)
    latent_dim = 64
    latent_length = 32  # sample_size / downsampling_ratio

    model = DeltaPredictor(
        input_dim=input_dim,
        latent_dim=latent_dim,
        latent_length=latent_length,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ダミー入力
    delta_f = torch.randn(batch_size, input_dim)
    latent1 = torch.randn(batch_size, latent_dim, latent_length)

    # 予測
    delta_g_pred = model(delta_f)
    print(f"Delta F shape: {delta_f.shape}")
    print(f"Delta G pred shape: {delta_g_pred.shape}")

    # 目標latent予測
    latent2_pred = model.predict_target_latent(latent1, delta_f)
    print(f"Latent2 pred shape: {latent2_pred.shape}")

    # 損失計算
    delta_g_true = torch.randn(batch_size, latent_dim, latent_length)
    criterion = DeltaPredictorLoss()
    losses = criterion(delta_g_pred, delta_g_true)
    print(f"\nLosses: {losses}")
