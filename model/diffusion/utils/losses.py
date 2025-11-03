from __future__ import print_function

import torch
from mediapipe.tasks.metadata.schema_py_generated import Tensor
from torch import nn
import torch.nn.functional as F
from scipy.spatial.distance import euclidean


def TemporalLoss(Y):
    diff = Y[:, 1:, :] - Y[:, :-1, :]
    t_loss = torch.mean(torch.norm(diff, dim=2, p=2) ** 2)
    return t_loss


def L1Loss(prediction, target, k=1, reduction="min", **kwargs):
    # prediction has shape of [batch_size, num_preds, features]
    # target has shape of [batch_size, num_preds, features]
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    # manual implementation of L1 loss
    loss = (torch.abs(prediction - target)).mean(axis=-1)

    # reduce across multiple predictions
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(axis=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


def MSELoss(prediction, target, k=1, reduction="mean", **kwargs):
    # prediction has shape of [batch_size, num_preds==k, features]
    # target has shape of [batch_size, num_preds==k, features]
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    # manual implementation of MSE loss
    loss = ((prediction - target) ** 2).mean(axis=-1)  # (batch_size, k)

    # reduce across multiple predictions
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(axis=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


def MSELossWithAct(prediction, target, k=1, reduction="mean", **kwargs):
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    bs, k, _ = prediction.shape
    prediction = prediction.reshape(bs, k, 50, 25)  # window_size==50, emotion_dim==25

    AU = prediction[:, :, :, :15]  # (bs, k, 50, 15)
    AU = torch.sigmoid(AU)

    middle_feat = prediction[:, :, :, 15:17]  # (bs, k, 50, 2)
    middle_feat = torch.tanh(middle_feat)

    emotion = prediction[:, :, :, 17:]  # (bs, k, 50, 8)
    emotion = torch.softmax(emotion, dim=-1)

    prediction = torch.cat((AU, middle_feat, emotion), dim=-1)
    prediction = prediction.reshape(bs, k, -1)
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"

    # manual implementation of MSE loss
    loss = ((prediction - target) ** 2).mean(axis=-1)  # (batch_size, k)

    # reduce across multiple predictions
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(axis=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


def KApproMSELoss(prediction, target, k, **kwargs):
    # prediction has shape of [batch_size, num_preds==k, features]
    # target has shape of [batch_size, num_preds==k, features]
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    bs, _, feat_dim = prediction.shape
    metrics = torch.zeros(size=(bs, 0, k)).to(prediction.device)
    preds = prediction.detach().clone()
    for idk in range(k):
        pred = preds[:, idk:idk + 1, :]  # (bs, 1, features)
        pred = pred.repeat(1, k, 1)  # (bs, k, features)
        mse = ((pred - target) ** 2).mean(axis=-1).unsqueeze(1)  # (bs, 1, k) each idk ==> all k
        metrics = torch.cat((metrics, mse), dim=1)
    # metrics.shape: (bs, k, k)
    minimum_mse = torch.argmin(metrics, dim=-1, keepdim=True)  # (bs, k, 1)
    minimum_mse = minimum_mse.repeat(1, 1, feat_dim).long()
    new_target = torch.gather(target, 1, minimum_mse)  # (bs, k, features)
    loss = MSELoss(prediction, new_target, k=1, reduction="mean")

    return loss


def SMLoss(prediction, target, k=1, **kwargs):
    """
    平滑损失
    """
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 4, "Only works with predictions of shape [batch_size, k, window_size, features]"
    sml1 = nn.SmoothL1Loss(reduction="mean")

    # [batch_size, k, window_size, features_dim=58]
    bs, k, window_size, _ = prediction.shape
    # 平滑损失
    loss = sml1((prediction[:, :, 2:, 52:] - prediction[:, :, 1:-1, 52:]),
                (prediction[:, :, 1:-1, 52:] - prediction[:, :, :-2, 52:])) + \
           0.5 * sml1(
        (prediction[:, :, 2:, :52] - prediction[:, :, 1:-1, :52]),
        (prediction[:, :, 1:-1, :52] - prediction[:, :, :-2, :52]))

    return loss

def gradient_loss(prediction, target, dim=2):
    """
    计算相邻时间点的梯度损失
    """
    pred_diff = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_diff = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.mse_loss(pred_diff, target_diff)


def SMLoss2(prediction, target, k=1,
            s_exp_w=1.0, s_r_w=5.0, s_t_w=10.0,
            s_mean_w=1.0, s_var_w=10.0,
            s_grad_w=1.0, **kwargs):
    """
    平滑损失
    """
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 4, "Only works with predictions of shape [batch_size, k, window_size, features]"
    sml1 = nn.SmoothL1Loss(reduction="mean")
    mse = nn.MSELoss(reduction="mean")

    # [batch_size, k, window_size, features_dim=58]
    bs, k, window_size, _ = prediction.shape
    # 平滑损失
    sm_loss_0_55 = sml1((prediction[:, :, 2:, :52] - prediction[:, :, 1:-1, :52]),
                        (prediction[:, :, 1:-1, :52] - prediction[:, :, :-2, :52]))

    sm_loss_52_55 = sml1((prediction[:, :, 2:, 52:55] - prediction[:, :, 1:-1, 52:55]),
                         (prediction[:, :, 1:-1, 52:55] - prediction[:, :, :-2, 52:55]))

    sm_loss_55_end = sml1((prediction[:, :, 2:, 55:] - prediction[:, :, 1:-1, 55:]),
                          (prediction[:, :, 1:-1, 55:] - prediction[:, :, :-2, 55:]))

    # 计算均值损失
    mean_pred = prediction[:, :, :, 55:-1].mean(dim=2)  # [batch_size, k, features]
    mean_target = target[:, :, :, 55:-1].mean(dim=2)  # [batch_size, k, features]
    mean_loss = mse(mean_pred, mean_target)

    # 计算方差损失
    var_pred = prediction[:, :, :, 55:].var(dim=2)  # [batch_size, k, features]
    var_target = target[:, :, :, 55:].var(dim=2)  # [batch_size, k, features]
    var_loss = mse(var_pred, var_target)

    # 计算梯度损失
    grad_loss = gradient_loss(prediction[:, :, :, 52:], target[:, :, :, 52:], dim=2)

    loss = (s_exp_w * sm_loss_0_55 +
            s_r_w * sm_loss_52_55 +
            s_t_w * sm_loss_55_end +
            s_mean_w * mean_loss +
            s_var_w * var_loss
            + s_grad_w * grad_loss
            )

    # print("sm_loss_0_55: ", sm_loss_0_55 * s_exp_w)
    # print("sm_loss_52_55: ", sm_loss_52_55 * s_r_w)
    # print("sm_loss_55_end: ", sm_loss_55_end * s_t_w)
    # print("mean_loss: ", mean_loss * s_mean_w)
    # print("var_loss: ", var_loss * s_var_w)
    # print("grad_loss: ", grad_loss * s_grad_w)
    #
    # print("loss: ", loss)

    # print("sm_loss_0_55_w: ", s_exp_w)
    # print("sm_loss_52_55_w: ", s_r_w)
    # print("sm_loss_55_end_w: ", s_t_w)
    # print("mean_loss_w: ", s_mean_w)
    # print("var_loss_w: ", s_var_w)
    # print("grad_loss_w: ", s_grad_w)


    return loss


def PosLoss(prediction, target, k=1,
            p_r_w=2.0,
            p_t_w=4.0,
            **kwargs):
    """
    额外姿态损失
    """
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 4, "Only works with predictions of shape [batch_size, k, window_size, features]"
    mse = nn.MSELoss(reduction="mean")

    # [batch_size, k, window_size, features_dim=58]
    bs, k, window_size, _ = prediction.shape
    # 姿态损失
    # 旋转
    pos_loss_52_55 = mse(prediction[:, :, :, 52:55], target[:, :, :, 52:55])

    # 平移（x, y）不包含深度
    pos_loss_55_57 = mse(prediction[:, :, :, 55:-1], target[:, :, :, 55:-1])

    loss = (pos_loss_52_55 * p_r_w
            + pos_loss_55_57 * p_t_w
            )

    # print("pos_loss_52_55: ", pos_loss_52_55 * pos_loss_52_55_w)
    # print("pos_loss_55_57: ", pos_loss_55_57 * pos_loss_55_57_w)
    # print("loss: ", loss)

    return loss


def OverlapLoss(pred, k=1, **kwargs):
    """
    重叠部分损失
    """
    # assert len(pred.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(pred.shape) == 4, "Only works with predictions of shape [batch_size, k, window_size, features]"
    overlap_loss = nn.MSELoss(reduction="mean")

    # [batch_size, k, window_size, features_dim=58]
    bs, k, two_window_size, _ = pred.shape
    window = two_window_size // 2
    # 裁切prediction和target
    prediction = pred[:, :, :window, :]
    target = pred[:, :, window:, :]

    loss = overlap_loss(prediction, target)

    return loss


def DiffusionLoss(
        output_prior,
        output_decoder,
        # ['KApproMSELoss', 'KApproMSELoss'] | [MSELoss, MSELoss] | [L1Loss, L1Loss] | ['...', 'MSELossWithAct']
        losses_type=['MSELoss', 'MSELoss'],  # MSELossWithAct for decoder training.
        losses_multipliers=[1, 1],
        losses_decoded=[False, True],
        k=10,  # k appropriate reactions
        temporal_loss_w=0.1,  # loss weight
        **kwargs):
    encoded_prediction = output_prior["encoded_prediction"].squeeze(-2)  # shape: (batch_size, k, encoded_dim)
    encoded_target = output_prior["encoded_target"].squeeze(-2)  # shape: (batch_size, k, encoded_dim)
    prediction_3dmm = output_decoder["prediction_3dmm"]  # shape: (batch_size, k, window_size, dim==58)
    target_3dmm = output_decoder["target_3dmm"]  # shape: (batch_size, k, window_size, dim==58)

    _, _, window_size, dim = prediction_3dmm.shape
    # compute losses
    losses_dict = {"loss": 0.0}

    losses_dict["temporal_loss"] = TemporalLoss(prediction_3dmm.reshape(-1, window_size, dim))
    losses_dict["loss"] += losses_dict["temporal_loss"] * temporal_loss_w
    assert temporal_loss_w <= 0.0, "we first disregard temporal loss."

    # join last two dimensions of prediction and target
    prediction_3dmm = prediction_3dmm.reshape(-1, k, window_size * dim)
    target_3dmm = target_3dmm.reshape(-1, k, window_size * dim)

    # reconstruction loss
    for loss_name, w, decoded in zip(losses_type, losses_multipliers, losses_decoded):
        # loss_final_name = loss_name + f"_{'decoded' if decoded else 'encoded'}"
        loss_final_name = f"{'decoded' if decoded else 'encoded'}"

        if decoded:
            losses_dict[loss_final_name] = eval(loss_name)(prediction_3dmm, target_3dmm, k=k)
        else:
            losses_dict[loss_final_name] = eval(loss_name)(encoded_prediction, encoded_target, k=k)

        losses_dict["loss"] += losses_dict[loss_final_name] * w

    return losses_dict


def DiffusionSmoothLoss(
        output_prior,
        output_decoder,
        # ['KApproMSELoss', 'KApproMSELoss'] | [MSELoss, MSELoss] | [L1Loss, L1Loss] | ['...', 'MSELossWithAct']
        losses_type=['MSELoss', 'MSELoss'],  # MSELossWithAct for decoder training.
        losses_multipliers=[1, 1],
        losses_decoded=[False, True],
        k=10,  # k appropriate reactions
        temporal_loss_w=0.1,  # loss weight
        **kwargs):
    encoded_prediction = output_prior["encoded_prediction"].squeeze(-2)  # shape: (batch_size, k, encoded_dim)
    encoded_target = output_prior["encoded_target"].squeeze(-2)  # shape: (batch_size, k, encoded_dim)
    prediction_3dmm = output_decoder["prediction_3dmm"]  # shape: (batch_size, k, window_size, dim==58)
    target_3dmm = output_decoder["target_3dmm"]  # shape: (batch_size, k, window_size, dim==58)

    _, _, window_size, dim = prediction_3dmm.shape
    # compute losses
    losses_dict = {"loss": 0.0}
    losses_dict["smooth_loss"] = SMLoss(prediction_3dmm, target_3dmm, k=k)
    losses_dict["loss"] += losses_dict["smooth_loss"]

    losses_dict["temporal_loss"] = TemporalLoss(prediction_3dmm.reshape(-1, window_size, dim))
    losses_dict["loss"] += losses_dict["temporal_loss"] * temporal_loss_w
    # assert temporal_loss_w <= 0.0, "we first disregard temporal loss."

    # join last two dimensions of prediction and target
    prediction_3dmm = prediction_3dmm.reshape(-1, k, window_size * dim)
    target_3dmm = target_3dmm.reshape(-1, k, window_size * dim)

    # reconstruction loss
    for loss_name, w, decoded in zip(losses_type, losses_multipliers, losses_decoded):
        # loss_final_name = loss_name + f"_{'decoded' if decoded else 'encoded'}"
        loss_final_name = f"{'decoded' if decoded else 'encoded'}"

        if decoded:
            losses_dict[loss_final_name] = eval(loss_name)(prediction_3dmm, target_3dmm, k=k)
        else:
            losses_dict[loss_final_name] = eval(loss_name)(encoded_prediction, encoded_target, k=k)

        losses_dict["loss"] += losses_dict[loss_final_name] * w

    return losses_dict


def DiffusionSmoothPosLoss(
        output_prior,
        output_decoder,
        # ['KApproMSELoss', 'KApproMSELoss'] | [MSELoss, MSELoss] | [L1Loss, L1Loss] | ['...', 'MSELossWithAct']
        losses_type=['MSELoss', 'MSELoss'],  # MSELossWithAct for decoder training.
        losses_multipliers=[1, 1],
        losses_decoded=[False, True],
        k=10,  # k appropriate reactions
        temporal_loss_w=0.1,  # loss weight
        smooth_loss_w=1.0,
        pos_loss_w=1.0,
        s_exp_w=1.0, s_r_w=5.0, s_t_w=10.0,
        s_mean_w=1.0, s_var_w=10.0, s_grad_w=1.0,
        p_r_w=2.0, p_t_w=4.0,
        **kwargs):
    encoded_prediction = output_prior["encoded_prediction"].squeeze(-2)  # shape: (batch_size, k, encoded_dim)
    encoded_target = output_prior["encoded_target"].squeeze(-2)  # shape: (batch_size, k, encoded_dim)
    prediction_3dmm = output_decoder["prediction_3dmm"]  # shape: (batch_size, k, window_size, dim==58)
    target_3dmm = output_decoder["target_3dmm"]  # shape: (batch_size, k, window_size, dim==58)

    _, _, window_size, dim = prediction_3dmm.shape
    # compute losses
    losses_dict = {"loss": 0.0}

    # 平滑损失
    # print("smooth_loss_w: ", smooth_loss_w)
    losses_dict["smooth_loss"] = SMLoss2(prediction_3dmm, target_3dmm, k=k,
                                                         s_exp_w=s_exp_w, s_r_w=s_r_w, s_t_w=s_t_w,
                                                         s_mean_w=s_mean_w, s_var_w=s_var_w, s_grad_w=s_grad_w)
    losses_dict["loss"] += losses_dict["smooth_loss"] * smooth_loss_w

    # 额外姿态损失
    losses_dict["pos_loss"] = PosLoss(prediction_3dmm, target_3dmm, k=k,
                                                   p_r_w=p_r_w, p_t_w=p_t_w)
    losses_dict["loss"] += losses_dict["pos_loss"] * pos_loss_w

    losses_dict["temporal_loss"] = TemporalLoss(prediction_3dmm.reshape(-1, window_size, dim))
    losses_dict["loss"] += losses_dict["temporal_loss"] * temporal_loss_w
    # assert temporal_loss_w <= 0.0, "we first disregard temporal loss."

    # join last two dimensions of prediction and target
    prediction_3dmm = prediction_3dmm.reshape(-1, k, window_size * dim)
    target_3dmm = target_3dmm.reshape(-1, k, window_size * dim)

    # reconstruction loss
    for loss_name, w, decoded in zip(losses_type, losses_multipliers, losses_decoded):
        # loss_final_name = loss_name + f"_{'decoded' if decoded else 'encoded'}"
        loss_final_name = f"{'decoded' if decoded else 'encoded'}"

        if decoded:
            losses_dict[loss_final_name] = eval(loss_name)(prediction_3dmm, target_3dmm, k=k)
        else:
            losses_dict[loss_final_name] = eval(loss_name)(encoded_prediction, encoded_target, k=k)

        losses_dict["loss"] += losses_dict[loss_final_name] * w

    return losses_dict


def TransLoss(overlap_pre, overlap_this, k=1, **kwargs):
    """
    重叠部分损失
    """
    # assert len(pred.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(overlap_pre.shape) == 4, "Only works with predictions of shape [batch_size, k, window_size, features]"

    # [batch_size, k, window_size, features_dim=58]
    bs, k, overlap_len, dim = overlap_pre.shape
    # 裁切prediction和target
    overlap_trans_pre = overlap_pre[:, :, :, 55:]
    overlap_tans_this = overlap_this[:, :, :, 55:]

    # Calculate linear weights
    steps = overlap_len
    weights = torch.linspace(0, 1, steps=steps).to(overlap_pre.device)

    # Expand weights to match the dimensions of A_disp and B_disp
    weights = weights.view(1, 1, -1, 1)  # Shape: (1, 1, steps, 1)

    # Compute the difference and apply the weights
    diff = (overlap_trans_pre - overlap_tans_this) ** 2  # Using L2 norm (squared difference)
    weights_diff = weights * diff

    # Compute the loss
    loss = weights_diff.mean()
    # print("trans_loss: ", loss)

    return loss


def DiffusionSmoothTransLoss(
        output_prior,
        output_decoder,
        # ['KApproMSELoss', 'KApproMSELoss'] | [MSELoss, MSELoss] | [L1Loss, L1Loss] | ['...', 'MSELossWithAct']
        losses_type=['MSELoss', 'MSELoss'],  # MSELossWithAct for decoder training.
        losses_multipliers=[1, 1],
        losses_decoded=[False, True],
        k=10,  # k appropriate reactions
        sliding_window=25,
        temporal_loss_w=0.1,  # loss weight
        trans_loss_w=1.0,
        smooth_loss_w=1.0,
        pos_loss_w=1.0,
        s_exp_w=1.0, s_r_w=5.0, s_t_w=10.0,
        s_mean_w=1.0, s_var_w=10.0, s_grad_w=1.0,
        p_r_w=2.0, p_t_w=4.0,
        **kwargs):
    encoded_prediction = output_prior["encoded_prediction"].squeeze(-2)  # shape: (batch_size, k, encoded_dim)
    encoded_target = output_prior["encoded_target"].squeeze(-2)  # shape: (batch_size, k, encoded_dim)
    prediction_3dmm = output_decoder["prediction_3dmm"]  # shape: (batch_size, k, window_size, dim==58)
    target_3dmm = output_decoder["target_3dmm"]  # shape: (batch_size, k, window_size, dim==58)

    _, _, window_size, dim = prediction_3dmm.shape
    # compute losses
    losses_dict = {"loss": 0.0}

    # 平滑损失
    losses_dict["smooth_loss"] = SMLoss(prediction_3dmm, target_3dmm, k=k)
    losses_dict["loss"] += losses_dict["smooth_loss"]

    if "prediction_3dmm_overlap_pre" in output_decoder:
        prediction_3dmm_overlap_pre = output_decoder[
            "prediction_3dmm_overlap_pre"]  # shape: (batch_size, k, window_size - sliding_size, dim==58)
        prediction_3dmm_overlap_this = output_decoder[
            "prediction_3dmm_overlap_this"]  # shape: (batch_size, k, window_size - sliding_size, dim==58)

        # 位移损失
        losses_dict["trans_loss"] = TransLoss(prediction_3dmm_overlap_pre, prediction_3dmm_overlap_this, k=k)
        # print("trans_loss: ",losses_dict["trans_loss"])
    else:
        # 将其转换为tensor
        losses_dict["trans_loss"] = torch.tensor(0.0).to(prediction_3dmm.device)
    losses_dict["loss"] += losses_dict["trans_loss"] * trans_loss_w

    losses_dict["temporal_loss"] = TemporalLoss(prediction_3dmm.reshape(-1, window_size, dim))
    losses_dict["loss"] += losses_dict["temporal_loss"] * temporal_loss_w
    # assert temporal_loss_w <= 0.0, "we first disregard temporal loss."

    # join last two dimensions of prediction and target
    prediction_3dmm = prediction_3dmm.reshape(-1, k, window_size * dim)
    target_3dmm = target_3dmm.reshape(-1, k, window_size * dim)

    # reconstruction loss
    for loss_name, w, decoded in zip(losses_type, losses_multipliers, losses_decoded):
        # loss_final_name = loss_name + f"_{'decoded' if decoded else 'encoded'}"
        loss_final_name = f"{'decoded' if decoded else 'encoded'}"

        if decoded:
            losses_dict[loss_final_name] = eval(loss_name)(prediction_3dmm, target_3dmm, k=k)
        else:
            losses_dict[loss_final_name] = eval(loss_name)(encoded_prediction, encoded_target, k=k)

        losses_dict["loss"] += losses_dict[loss_final_name] * w

    return losses_dict
