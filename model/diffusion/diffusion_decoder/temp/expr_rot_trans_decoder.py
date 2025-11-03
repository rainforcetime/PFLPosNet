import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.ReLU()):
        super(MLP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(dims[-2], dims[-1]))  # No activation on the last layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ExprRotTransDecoder(nn.Module):
    def __init__(self, latent_dim, nfeats, expr_dim, rot_trans_dim):
        super(ExprRotTransDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.expr_dim = expr_dim
        self.rot_trans_dim = rot_trans_dim

        self.to_3dmm_expr_feat = nn.Linear(latent_dim, expr_dim) if latent_dim != nfeats else nn.Identity()
        self.to_3dmm_rot_feat = nn.Linear(latent_dim + expr_dim,
                                          rot_trans_dim // 2) if latent_dim != nfeats else nn.Identity()
        self.to_3dmm_trans_feat = nn.Linear(latent_dim + expr_dim,
                                            rot_trans_dim // 2) if latent_dim != nfeats else nn.Identity()

    def forward(self, sample):
        """
        :param sample: (bs * k, sel_len, latent_dim)
        :return: (bs * k, sel_len, expr_dim+rot_trans_dim)
        """
        expr_sample = self.to_3dmm_expr_feat(sample)
        concat_sample_expr = torch.cat((sample, expr_sample), dim=-1)
        rot_sample = self.to_3dmm_rot_feat(concat_sample_expr)
        trans_sample = self.to_3dmm_trans_feat(concat_sample_expr)

        sample = torch.cat((expr_sample, rot_sample, trans_sample), dim=-1)

        return sample


class ExprRotTransDecoder_v2(nn.Module):
    def __init__(self, latent_dim, nfeats, expr_dim, rot_trans_dim, expr_hidden_dims=None, rot_trans_hidden_dim=None):
        super(ExprRotTransDecoder_v2, self).__init__()
        self.latent_dim = latent_dim
        self.expr_dim = expr_dim
        self.rot_trans_dim = rot_trans_dim

        # Define a more complex MLP for expression coefficients
        if expr_hidden_dims is None:
            expr_hidden_dims = [256, 128, 64]  # Complex hidden layers dimensions for expression MLP
        self.to_3dmm_expr_feat = MLP(latent_dim, expr_dim, expr_hidden_dims) if latent_dim != nfeats else nn.Identity()
        # Define simpler MLPs for rotation and translation coefficients
        rot_trans_input_dim = latent_dim + expr_dim
        if rot_trans_hidden_dim is None:
            rot_trans_hidden_dim = 64  # Simple hidden layer dimension for rotation and translation MLPs
        self.to_3dmm_rot_feat = MLP(rot_trans_input_dim, rot_trans_dim // 2,
                                    [rot_trans_hidden_dim]) if latent_dim != nfeats else nn.Identity()
        self.to_3dmm_trans_feat = MLP(rot_trans_input_dim, rot_trans_dim // 2,
                                      [rot_trans_hidden_dim]) if latent_dim != nfeats else nn.Identity()

    def forward(self, sample):
        """
        :param sample: (bs * k, sel_len, latent_dim)
        :return: (bs * k, sel_len, expr_dim+rot_trans_dim)
        """
        expr_sample = self.to_3dmm_expr_feat(sample)
        concat_sample_expr = torch.cat((sample, expr_sample), dim=-1)
        rot_sample = self.to_3dmm_rot_feat(concat_sample_expr)
        trans_sample = self.to_3dmm_trans_feat(concat_sample_expr)

        sample = torch.cat((expr_sample, rot_sample, trans_sample), dim=-1)

        return sample
