import torch
import torch.nn.functional as F
from nms.training.process import get_dest_frame
from nms.modules.discriminator import Discriminator

'''
Loss functions used for motion generation
'''
def trans_loss(
    out: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor
):
    loss_trans = F.l1_loss(out, gt, reduction="none")
    loss_trans = mask_loss(loss_trans, mask)
    return loss_trans

def fk_loss(
    pred_world_pos: torch.Tensor,
    gt_world_pos: torch.Tensor,
    mask: torch.Tensor
):
    loss_fk = F.l1_loss(pred_world_pos, gt_world_pos, reduction="none")
    loss_fk = loss_fk.reshape([*loss_fk.shape[:-2], -1])
    loss_fk = mask_loss(loss_fk, mask)
    return loss_fk

def dest_loss(
    out: torch.Tensor,
    gt: torch.Tensor,
    feat_start_idx: dict,
    feat_end_idx: dict,
    valid_nof: torch.Tensor,
    block_nof: int
):
    out_dest_frame = get_dest_frame(out, valid_nof, block_nof) # make network predict first root trf for targetB
    gt_dest_frame = get_dest_frame(gt, valid_nof, block_nof)
    out_dest_root = out_dest_frame[..., feat_start_idx["root_height"]:feat_end_idx["root_ang_disp"]] # this assumes root features start with root height and ends with root ang disp
    gt_dest_root = gt_dest_frame[..., feat_start_idx["root_height"]:feat_end_idx["root_ang_disp"]]
    loss_dest = F.l1_loss(out_dest_root, gt_dest_root)
    return loss_dest

def adv_loss(
    disc: Discriminator,
    disc_fake_input: torch.Tensor,
    disc_loss_mask: torch.Tensor
):
    fake_logit, _ = disc(disc_fake_input)
    fake_label = torch.ones_like(fake_logit, device=fake_logit.device)

    fake_logit = fake_logit * disc_loss_mask
    fake_label = fake_label * disc_loss_mask
    loss_adv = 0.5 * F.mse_loss(fake_logit, fake_label)
    return loss_adv

def disc_loss(
    gt_world_pos: torch.Tensor,
    disc: Discriminator,
    disc_fake_input: torch.Tensor,
    disc_loss_mask: torch.Tensor,
    block_nof: int,
    mode: str = "train"
):
    half = block_nof // 2
    gt_trans_disp = gt_world_pos[:, 1:, disc.jnt_map] - gt_world_pos[:, :-1, disc.jnt_map]
    gt_trans_disp_trim = gt_trans_disp[:, half:-half+1]
    gt_trans_disp_trim = torch.cat([gt_trans_disp_trim, gt_trans_disp[:, :block_nof]], dim=1)
    real_input = gt_trans_disp_trim.reshape([-1, block_nof, gt_trans_disp_trim.shape[-2] * gt_trans_disp_trim.shape[-1]])

    disc_fake_input = disc_fake_input.detach()
    real_input = real_input.detach().requires_grad_(True) # needed for gradient penalty

    fake_logit, _ = disc(disc_fake_input)
    real_logit, real_input_norm = disc(real_input)
    if mode == "val":
        disc_loss_dict = disc.compute_loss(fake_logit, real_logit, disc_loss_mask)
    else:
        disc_loss_dict = disc.compute_loss(fake_logit, real_logit, disc_loss_mask, real_input_norm)
    return disc_loss_dict

def mask_loss(
    loss: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    '''
    Mask the computed loss

    Args:
        loss: [batch, nof, n_feat]
    '''
    mask_expanded = mask.unsqueeze(-1)
    masked_loss = loss * mask_expanded.float()
    valid_loss = masked_loss.mean(dim=-1) # loss per frame
    valid_loss_sum = valid_loss.sum(dim=-1) # loss sum per motion

    # pred_nof = valid_nof - 2 * block_nof # exclude number of frames in the given motion
    pred_nof = mask.sum(dim=-1)
    if reduction == "mean":
        final_loss = (valid_loss_sum / pred_nof).mean()
    elif reduction == "sum":
        final_loss = (valid_loss_sum / pred_nof).sum()
    return final_loss