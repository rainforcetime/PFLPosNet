import torch
import torch.nn as nn


class CouplingLayer(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(CouplingLayer, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features

        # 定义一个简单的全连接网络，用来生成缩放因子
        self.mlp = nn.Sequential(
            nn.Linear(in_features // 2, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, in_features // 2)
        )

    def forward(self, x, reverse=False):
        # 将输入x拆分为两部分
        x1, x2 = x.chunk(2, dim=-1)

        if not reverse:
            # 正向传递：x2变换为x2'，同时应用缩放因子
            log_scale = self.mlp(x1)  # 生成缩放因子
            y2 = x2 * torch.exp(log_scale) + x1
            return torch.cat([x1, y2], dim=-1), log_scale
        else:
            # 反向传递：将x2'还原为x2
            log_scale = self.mlp(x1)
            y2 = (x2 - x1) * torch.exp(-log_scale)
            return torch.cat([x1, y2], dim=-1)


class InvertibleNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(InvertibleNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 正向投影层，将58维映射到hidden_dim维度
        self.proj_in_forward = nn.Linear(input_dim, hidden_dim)  # 正向输入映射 (58 -> 256)
        # 反向投影层，将512维映射到hidden_dim维度
        self.proj_in_inverse = nn.Linear(output_dim, hidden_dim)  # 反向输入映射 (512 -> 256)

        # 输出投影层，将hidden_dim映射到最终的维度
        self.proj_out = nn.Linear(hidden_dim, output_dim)  # 输出映射 (256 -> 512)

        # 构建多个耦合层
        self.layers = nn.ModuleList([CouplingLayer(hidden_dim, hidden_dim) for _ in range(n_layers)])

    def forward(self, x):
        log_det_jacobian = 0
        B, T, dim = x.shape

        # 将输入从 (B, T, dim) 转换为 (B*T, dim) 以便与线性层兼容
        x = x.view(-1, dim)  # 扁平化：B*T, dim

        # 输入投影：将 58 映射到 hidden_dim（例如 256）
        x = self.proj_in_forward(x)

        # 将其形状恢复回 (B, T, hidden_dim)
        x = x.view(B, T, self.hidden_dim)

        for layer in self.layers:
            # 对每个时间步的样本分别进行变换
            x, log_scale = layer(x, reverse=False)
            log_det_jacobian += log_scale.sum(dim=-1)

        # 输出投影：将 hidden_dim 映射到 512
        x = self.proj_out(x)

        return x, log_det_jacobian

    def inverse(self, x):
        B, T, dim = x.shape

        # 将输入从 (B, T, dim) 转换为 (B*T, dim) 以便与线性层兼容
        x = x.view(-1, dim)  # 扁平化：B*T, dim

        # 输入反投影：将 512 映射回 hidden_dim（256）
        x = self.proj_in_inverse(x)

        # 将其形状恢复回 (B, T, hidden_dim)
        x = x.view(B, T, self.hidden_dim)

        for layer in reversed(self.layers):
            # 对每个时间步的样本分别进行反向变换
            x = layer(x, reverse=True)

        # 输出反投影：将 hidden_dim 映射回 58
        x = self.proj_out(x)

        return x


if __name__ == '__main__':
    # 示例用法：
    input_dim = 58  # 输入的特征维度
    output_dim = 512  # 输出的特征维度
    hidden_dim = 256  # 隐藏层维度
    n_layers = 6  # 网络层数

    # 创建模型
    model = InvertibleNetwork(input_dim=output_dim, hidden_dim=hidden_dim, n_layers=n_layers)

    # 创建随机输入数据 (B, T, input_dim)
    batch_size = 20
    T = 50  # 时间步数
    x = torch.randn(batch_size, T, input_dim)

    # 正向传递 (58 -> 512)
    y, log_det_jacobian = model.forward(x)
    print(f"Output shape (58 -> 512): {y.shape}")

    # 反向传递 (512 -> 58)
    x_reconstructed = model.inverse(y)
    print(f"Reconstructed shape (512 -> 58): {x_reconstructed.shape}")
