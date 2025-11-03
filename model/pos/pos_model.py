import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.diffusion.depth import DepthDetector


class PosModifierMLP(nn.Module):
    def __init__(self, cfg):
        super(PosModifierMLP, self).__init__()
        self.window_size = cfg.window_size
        self.slide_size = cfg.slide_size
        self.hidden_dim = cfg.hidden_dim
        self.pos_dim = cfg.pos_dim
        self.continuity_loss_w = cfg.continuity_loss_w
        self.sm_loss_w = cfg.sm_loss_w
        self.trans_smoothness_w = cfg.trans_smoothness_w
        self.second_diff_w = cfg.second_diff_w
        self.connect_spread_size = cfg.connect_spread_size
        self.connect_loss_w = cfg.connect_loss_w

        # Encoder for current_non_overlap
        self.encoder = nn.Sequential(
            nn.Conv1d(self.pos_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling to get a single vector per sequence
        )

        # Define the MLP layers for decoding
        self.fc1 = nn.Linear(self.pos_dim + self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.slide_size * self.pos_dim)

    def forward(self, past_pos, current_pos, target_pos=None):
        """
        :param past_pos: (B, K, window_size, pos_dim)
        :param current_pos: (B, K, window_size, pos_dim)
        :return:
        """
        B, K, _, pos_dim = past_pos.size()

        # Slice past overlap and current non-overlap
        past_overlap = past_pos[:, :, -self.window_size + self.slide_size:,
                       :]  # (B, K, window_size - slide_size, pos_dim)
        current_non_overlap = current_pos[:, :, -self.slide_size:, :]  # (B, K, slide_size, pos_dim)

        # Extract the end of past overlap
        past_end = past_overlap[:, :, -1, :]  # (B, K, pos_dim)

        # Reshape current_non_overlap for the encoder
        encoded_current = self.encode_sequence(current_non_overlap)  # (B, K, hidden_dim)

        # Combine the end of past overlap with the encoded current_non_overlap features
        combined_input = torch.cat((past_end, encoded_current), dim=-1)  # (B, K, pos_dim + hidden_dim)

        # Pass through the MLP to generate new non-overlapping part
        x = F.relu(self.fc1(combined_input))
        x = F.relu(self.fc2(x))
        new_current_non_overlap_flat = self.fc3(x)  # (B, K, slide_size * pos_dim)

        # Reshape to get the correct dimensions
        new_current_non_overlap = new_current_non_overlap_flat.view(B, K, self.slide_size,
                                                                    pos_dim)  # (B, K, slide_size, pos_dim)

        # Concatenate the past overlap with the newly generated non-overlapping part
        output = torch.cat((past_overlap, new_current_non_overlap), dim=2)  # (B, K, window_size, pos_dim)

        # Compute the loss if target_pos is provided
        loss = None
        if target_pos is not None:
            loss = self.compute_smoothness_loss(past_overlap, new_current_non_overlap, output)

        return output, loss

    def encode_sequence(self, seq):
        """
        Encodes the sequence using the encoder.
        :param seq: (B, K, slide_size, pos_dim)
        :return: (B, K, hidden_dim)
        """
        B, K, slide_size, pos_dim = seq.size()
        # Reshape to (B * K, pos_dim, slide_size) for convolution
        seq_reshaped = seq.permute(0, 1, 3, 2).reshape(-1, pos_dim, slide_size)  # (B * K, pos_dim, slide_size)

        # Encode the sequence
        encoded = self.encoder(seq_reshaped).squeeze(-1)  # (B * K, hidden_dim)

        # Reshape back to (B, K, hidden_dim)
        encoded_seq = encoded.view(B, K, -1)  # (B, K, hidden_dim)

        return encoded_seq

    def compute_smoothness_loss(self, past_overlap, predicted_current, output):
        """
        Computes smoothness and continuity losses to ensure the predicted positions are continuous and smooth.
        """
        # 鼓励拼接位置前后的连续性，即拼接位置前后的位置应该相近
        continuity_loss = torch.mean(torch.abs(past_overlap[:, :, -1, :] - predicted_current[:, :, 0, :]))

        # 一阶差分损失 鼓励整体序列中相邻时间步的变化幅度较小
        smoothness_loss = torch.mean(torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :]))

        start_loc = self.slide_size - 1
        end_loc = start_loc + self.connect_spread_size

        # 拼接处spread_size的二阶差分损失，鼓励拼接处的平滑过渡，多个时间步的变化幅度相近
        connect_diff1 = output[:, :, start_loc:end_loc, :] - output[:, :, start_loc-1:end_loc-1, :]
        connect_loss = torch.mean(torch.abs(connect_diff1[:, :, 1:] - connect_diff1[:, :, :-1]))

        # 二阶差分损失 鼓励整体序列的变化幅度相近
        diff1 = output[:, :, 1:] - output[:, :, :-1]
        diff2 = torch.mean(torch.abs(diff1[:, :, 1:] - diff1[:, :, :-1]))

        # 可选）鼓励从 past_overlap 平滑过渡到 new_current_non_overlap （）
        transition_smoothness_loss = 0
        if self.trans_smoothness_w > 0:
            # Calculate the difference between the last few steps of past_overlap and the first few steps of predicted_current
            transition_smoothness_loss = torch.mean(torch.abs(
                past_overlap[:, :, -self.slide_size:, :] - predicted_current[:, :, :self.slide_size, :]
            ))

        # print("continuity_loss: ", continuity_loss)
        # print("smoothness_loss: ", smoothness_loss)
        # print("transition_smoothness_loss: ", transition_smoothness_loss)

        return (
                self.continuity_loss_w * continuity_loss +
                self.sm_loss_w * smoothness_loss +
                self.trans_smoothness_w * transition_smoothness_loss
                + self.second_diff_w * diff2
                + self.connect_loss_w * connect_loss
        )

    def get_model_name(self):
        return self.__class__.__name__


class PosNet(nn.Module):  # a modify net
    def __init__(self, cfg, main_net, device):
        super().__init__()
        self.mode = cfg.mode # train (val) or test
        cfg_this = cfg.pos_model.args
        self.main_net = main_net  # instance of our diffusion model
        self.freeze_main_net = cfg_this.freeze_main_net  # freeze the main net or not
        self.slide_size = cfg_this.slide_size

        if self.freeze_main_net:
            for param in self.main_net.parameters():
                param.requires_grad = False

        self.device = device
        self.pos_dim = cfg_this.pos_dim
        self.window_size = cfg_this.window_size
        self.use_detect_depth = cfg_this.get("use_detect_depth", False)

        self.posnet = PosModifierMLP(
            cfg_this
        ).to(device)

        is_resume = cfg.pos_model.args.get("resume")
        # Load posnet_path for inference
        if is_resume is not None:
            model_path = cfg.pos_model.args.get("resume")
            assert model_path is not None, "The model path should be provided."
            model_name = self.posnet.get_model_name()
            save_path = os.path.join(cfg.trainer.saving_checkpoint_dir, model_name, model_path)
            assert os.path.exists(save_path), "Miss checkpoint for posnet: {}.".format(save_path)
            checkpoint = torch.load(save_path, map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']
            self.posnet.load_state_dict(state_dict)
            print("Successfully load model for inference: {}, {}".format(model_name, model_path))

        # Load depth detector for inference
        if self.use_detect_depth:
            depth_cfg = cfg.depth_detector.args
            self.depth_detector = DepthDetector(depth_cfg)

            # load checkpoint
            checkpoint = torch.load(depth_cfg.checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            self.depth_detector.load_state_dict(state_dict)
            print("Depth detector loaded from: ", depth_cfg.checkpoint_path)
            self.depth_detector.to(device)
            self.depth_detector.eval()

    def forward(self, x):
        if self.mode in ['train', 'val']:
            prior, output = self.main_net(**x) # (batch_size, k, window_size, dim=58)
            prediction_3dmm = output["prediction_3dmm"]
            B, K, window_size, _3dmm_dim = prediction_3dmm.shape
            target_3dmm = output["target_3dmm"]
            past_listener_3dmmenc = output["past_listener_3dmmenc"].reshape(B, K, window_size, _3dmm_dim)

            past_pos = past_listener_3dmmenc[:, :, :, -self.pos_dim:]
            current_pos = prediction_3dmm[:, :, :, -self.pos_dim:]
            target_pos = target_3dmm[:, :, :, -self.pos_dim:]
            prediction_3dmm[:, :, :, -self.pos_dim:], sm_loss = self.posnet(past_pos, current_pos, target_pos)

            output["prediction_3dmm"] = prediction_3dmm

            return prior, output, sm_loss

        else:
            # inference
            listener_reference = x["listener_reference"]
            prior, output = self.main_net(**x) # (batch_size, k, n, window_size, dim=58)
            prediction_3dmm = output["prediction_3dmm"]
            B, K, N, window_size, _3dmm_dim  = prediction_3dmm.shape
            past_listener_3dmmenc = torch.zeros(
                size=(B, K, window_size, _3dmm_dim)
            ).to(device=prediction_3dmm.device)
            _output_listener_3dmm = (torch.randn(size=(B, K, 0, _3dmm_dim))
                                     .to(prediction_3dmm.device))
            _output_listener_3dmm_old = (torch.randn(size=(B, K, 0, _3dmm_dim))
                                     .to(prediction_3dmm.device))
            total_loss = torch.tensor(0.0).to(device=prediction_3dmm.device)

            for i in range(N):
                if i == 0:
                    # past_listener_3dmmenc = output["past_listener_3dmmenc"]
                    # prediction_3dmm[:, :, i] = self.pos_net(past_listener_3dmmenc, prediction_3dmm[:, :, i])
                    _output_listener_3dmm = torch.cat((_output_listener_3dmm,
                                                       prediction_3dmm[:, :, i, :, :]), dim=-2)

                    _output_listener_3dmm_old = torch.cat((_output_listener_3dmm_old,
                                                       prediction_3dmm[:, :, i, :, :]), dim=-2)
                    past_listener_3dmmenc = prediction_3dmm[:, :, i, :, :]
                else:
                    past_pos = past_listener_3dmmenc[:, :, :, -self.pos_dim:]
                    current_pos = prediction_3dmm[:, :, i, :, -self.pos_dim:]
                    _output_listener_3dmm_old = torch.cat((_output_listener_3dmm_old,
                                                           prediction_3dmm[:, :, i, -self.slide_size:, :]), dim=-2)

                    prediction_3dmm[:, :, i, :, -self.pos_dim:], _ = self.posnet(past_pos, current_pos)

                    _output_listener_3dmm = torch.cat((_output_listener_3dmm,
                                                       prediction_3dmm[:, :, i, -self.slide_size:, :]), dim=-2)

                    past_listener_3dmmenc = prediction_3dmm[:, :, i, :, :]
                # print("i: ", i)
                # print("_output_listener_3dmm len: ", _output_listener_3dmm.shape[-2])

            if self.use_detect_depth:
                with torch.no_grad():
                    reference_depth = self.depth_detector.get_depth(listener_reference)
                print("reference_depth: ", reference_depth.shape)
                first_time_step = _output_listener_3dmm[:, :, 0, -1].unsqueeze(-1)  # 形状变为 (B, K, 1)
                _output_listener_3dmm[:,:,:,-1] -= first_time_step
                _output_listener_3dmm[:,:,:,-1] += reference_depth

            output["prediction_3dmm"] = _output_listener_3dmm
            output["prediction_3dmm_old"] = _output_listener_3dmm_old
            output["prediction_emotion_old"]  = self.main_net.diffusion_decoder.latent_3dmm_embedder.coeff_reg(output["prediction_3dmm_old"])
            output["prediction_emotion"] = self.main_net.diffusion_decoder.latent_3dmm_embedder.coeff_reg(output["prediction_3dmm"])

            return prior, output, total_loss

    def get_model_name(self):
        return self.__class__.__name__
