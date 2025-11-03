import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseTransformationMLP(nn.Module):
    def __init__(self, latent_dim=512, pos_dim=6, nfeat=58, hidden_dims=[256, 128]):
        super(PoseTransformationMLP, self).__init__()

        # 定义一个简单的MLP来映射past_pos到latent_dim维度
        self.pos_mapping = nn.Sequential(
            nn.Linear(pos_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # 定义用于拼接后数据的多层MLP
        concat_dim = latent_dim * 2  # 因为我们将sample和mapped_past_pos在最后一个维度拼接
        layers = []
        input_dim = concat_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim

        # 最后一层映射到nfeat维度
        layers.append(nn.Linear(input_dim, nfeat))

        self.mlp = nn.Sequential(*layers)

        # 初始化参数以确保稳定性（可选）
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, sample, past_pos):
        # 映射past_pos到latent_dim
        mapped_past_pos = self.pos_mapping(past_pos)

        # 拼接sample与mapped_past_pos
        concatenated = torch.cat((sample, mapped_past_pos), dim=-1)

        # 经过多层mlp映射到nfeat维度
        output = self.mlp(concatenated)

        # # 分割输出为表情系数和位姿系数
        # expression_coeffs, pose_coeffs = torch.split(output, [52, 6], dim=-1)
        #
        # # 应用激活函数或其他方法保证表情系数多样性，位姿系数稳定
        # # 这里可以考虑对pose_coeffs应用某种约束或正则化
        # # 对于expression_coeffs，可以不进行额外处理或者应用适当的激活函数
        #
        # # 在这里简单地将两个部分重新组合
        # output = torch.cat((expression_coeffs, pose_coeffs), dim=-1)

        return output


if __name__ == '__main__':
    # 假设B=32, K=10, T=20作为示例batch size, keypoint number, 和temporal length
    model = PoseTransformationMLP()

    # 示例输入
    B, K, T = 32, 10, 20
    sample = torch.randn(B, K, T, 512)  # 高维变量
    past_pos = torch.randn(B, K, T, 6)  # 过去窗口的位姿信息

    # 获取模型输出
    output = model(sample, past_pos)

    print(output.shape)  # 应该打印 (32, 10, 20, 58)