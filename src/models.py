import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        weight_decay: float = 0.001
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)
    
    def l2_regularization(self):
        l2_reg = torch.tensor(0., device=next(self.parameters()).device)  # 初期化
        for param in self.parameters():
            l2_reg += torch.norm(param)
        return l2_reg

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)


class EEGNet(nn.Module):
    def __init__(self, F1=16, F2=32, D=2, dropout_rate=0.25):
        super(EEGNet, self).__init__()
        
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), stride=(1, 1), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(F1, F1*D, kernel_size=(2, 1), stride=(1, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=dropout_rate),
        )
        
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1*D, F2, kernel_size=(1, 16), stride=(1, 1), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=dropout_rate),
        )
        
        self.classify = nn.Sequential(
            nn.Linear(F2*(((271 - 1) // 4 + 1 - 1) // 8 + 1), 4),  # Assuming 4 output classes
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        if x.dim() == 3:  # チャンネル次元がない場合は追加
            x = x.unsqueeze(1)
        x = self.first_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x