import torch
import torch.nn as nn

class PoseGenerator(nn.Module):
    def __init__(self, window_size, slide_size, input_dim=512,hidden_size=128, pose_dim=6):
        super(PoseGenerator, self).__init__()
        self.window_size = window_size
        self.slide_size = slide_size
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.pose_dim = pose_dim
        self.lstm = nn.LSTM(input_dim + pose_dim, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, pose_dim * slide_size)

    def forward(self, last_pose, influence_factor):
        """
        :param last_pose: (B * K, window_size, pose_dim)
        :param influence_factor:  (B * K, window_size, latent_dim)
        :return:
        """
        combined_input = torch.cat([last_pose, influence_factor], dim=-1)
        lstm_output, _ = self.lstm(combined_input.view(-1, self.window_size, self.input_dim + self.pose_dim))
        output = self.fc(lstm_output[:, -1, :]) # 取最后一个时间步的输出作为预测
        # print(output.shape)

        return output.view(-1, self.slide_size, self.pose_dim)