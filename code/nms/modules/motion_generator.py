import torch
from nms.modules.transformer import Transformer
from nms.motion.motion_op_tensor import fk
import nms.utils.bvh as bvh
import nms.utils.utils as utils
from nms.rotation.rotation_tensor import ortho6d_to_rotmat, rotmat_to_trf
from nms.training.process_tensor import to_trf, align, find_closest_trf_to
from nms.training.process import get_last_valid_root_trf, get_dest_frame, get_last_valid_block
import argparse

class MotionGenerator:
    def __init__(
        self,
        config: argparse.Namespace, # train config
        device: torch.device,
        mode: str = "train"
    ):
        self.config = config
        self.device = device
        self.skel_path = config.skel_path
        self.skel = bvh.load(self.skel_path, load_motion=False).skel

        if mode == "train":
            stats_dir = f"{config.dataset_dir}/{config.dataset_name}"
            datarep_dir = f"{config.dataset_dir}/{config.dataset_name}"
            datagen_config = utils.load_config(datarep_dir, "mg_datagen")
            out_datarep_dir = "./nms/datarep"
            out_datarep_name = config.out_datarep_name
            self.block_nof = datagen_config.block_nof
        else: # validation
            stats_dir = "" # included in state_dict
            datarep_dir = f"{config.result_dir}/{config.exp_name}"
            out_datarep_dir = datarep_dir
            out_datarep_name= "out_datarep"
            self.block_nof = config.block_nof

        # load datarep
        self.target_datarep = utils.load_config(datarep_dir, "target_datarep", to_dict=True)
        self.guide_datarep = utils.load_config(datarep_dir, "guide_datarep", to_dict=True)
        self.gt_datarep = utils.load_config(datarep_dir, "gt_datarep", to_dict=True)
        self.out_datarep = utils.load_config(out_datarep_dir, out_datarep_name, to_dict=True)

        self.target_total_dim, self.target_feat_start_idx, self.target_feat_end_idx, _, _ = utils.get_feature_dims(self.target_datarep)
        self.guide_total_dim, self.guide_feat_start_idx, self.guide_feat_end_idx, _, _ = utils.get_feature_dims(self.guide_datarep)
        _, self.gt_feat_start_idx, self.gt_feat_end_idx, _, _ = utils.get_feature_dims(self.gt_datarep)
        self.out_total_dim, self.out_feat_start_idx, self.out_feat_end_idx, _, _ = utils.get_feature_dims(self.out_datarep)

        # set other vars
        self.config.block_nof = self.block_nof
        self.max_len = config.lengths[-1] * self.block_nof

        # init transformer
        stats_path = f"{stats_dir}/statistics.pt"
        input_dim = self.target_total_dim + self.guide_total_dim
        output_dim = self.out_total_dim
        self.config.archi.input_dim = input_dim
        self.config.archi.out_dim = output_dim
        self.config.archi.max_len = self.max_len
        self.transformer = Transformer(self.config.archi, stats_path).to(device)

    def save_datarep(self, save_dir):
        utils.save_config(self.target_datarep, save_dir, "target_datarep")
        utils.save_config(self.guide_datarep, save_dir, "guide_datarep")
        utils.save_config(self.gt_datarep, save_dir, "gt_datarep")
        utils.save_config(self.out_datarep, save_dir, "out_datarep")

    def get_aligned_world_positions(
        self,
        input: torch.Tensor,
        out: torch.Tensor,
        valid_nof: torch.Tensor,
        padding_mask: torch.Tensor,
        gt: torch.Tensor = None
    ):
        world_trfs = self.get_aligned_world_trfs(input, out, valid_nof, padding_mask=padding_mask, gt=gt)
        world_pos = {}
        for key in world_trfs:
            world_pos[key] = world_trfs[key][..., :3, 3]
        return world_pos

    def get_aligned_world_trfs(
        self,
        input: torch.Tensor,
        out: torch.Tensor,
        valid_nof: torch.Tensor,
        padding_mask: torch.Tensor,
        gt: torch.Tensor = None
    ):
        block_nof = self.block_nof
        target = input[..., :self.target_total_dim]

        # accumulate vals for world position
        pred_feat = torch.cat([target[:, :block_nof], out[:, block_nof:]], dim=1) # replace first block with target
        pred_local_trf = to_trf(pred_feat, self.gt_feat_start_idx, self.gt_feat_end_idx, self.skel, padding_mask=padding_mask)
        pred_world_trf = fk(pred_local_trf, self.skel)

        # align targetB and get world pos
        targetB_trf_aligned = self.align_targetB(target, pred_world_trf, valid_nof)
        targetB_world_trf = fk(targetB_trf_aligned, self.skel)

        world_trfs = {
            "targetB": targetB_world_trf,
            "pred": pred_world_trf
        }

        if gt is not None:
            gt_local_trf = to_trf(gt, self.gt_feat_start_idx, self.gt_feat_end_idx, self.skel, padding_mask=padding_mask)
            gt_world_trf = fk(gt_local_trf, self.skel)
            world_trfs["gt"] = gt_world_trf

        return world_trfs

    def get_aligned_local_trfs(
        self,
        input: torch.Tensor,
        out: torch.Tensor,
        valid_nof: torch.Tensor,
        padding_mask: torch.Tensor,
        gt: torch.Tensor = None
    ):
        block_nof = self.block_nof
        target = input[..., :self.target_total_dim]

        # accumulate vals for world position
        pred_feat = torch.cat([target[:, :block_nof], out[:, block_nof:]], dim=1) # replace first block with target
        pred_local_trf = to_trf(pred_feat, self.gt_feat_start_idx, self.gt_feat_end_idx, self.skel, padding_mask=padding_mask)
        targetB_trf_aligned = self.align_targetB(target, pred_local_trf, valid_nof)

        local_trfs = {
            "targetB": targetB_trf_aligned,
            "pred": pred_local_trf
        }

        if gt is not None:
            gt_local_trf = to_trf(gt, self.gt_feat_start_idx, self.gt_feat_end_idx, self.skel, padding_mask=padding_mask)
            local_trfs["gt"] = gt_local_trf
        return local_trfs

    def get_align_trf(
        self,
        out,
        valid_nof
    ):
        out_align_vec = get_dest_frame(out, valid_nof, self.block_nof)[:, 0]
        translate = out_align_vec[..., self.gt_feat_start_idx["root_trans_disp"]:self.gt_feat_end_idx["root_trans_disp"]]
        align_translate = torch.stack([translate[..., 0], torch.tensor(0).to(self.device), translate[..., 2]], dim=0)

        rotate = out_align_vec[..., self.gt_feat_start_idx["root_orient"]:self.gt_feat_end_idx["root_orient"]]
        rotate_rotmat = ortho6d_to_rotmat(rotate)
        align_rotmat_col0 = torch.stack([rotate_rotmat[..., 0, 0], torch.tensor(0).to(self.device), rotate_rotmat[..., 0, 2]], dim=0)
        align_rotmat_col1 = torch.tensor([0, 1, 0]).to(self.device)
        align_rotmat_col2 = torch.stack([rotate_rotmat[..., 0, 2], torch.tensor(0).to(self.device), rotate_rotmat[..., 2, 2]], dim=0)
        align_rotmat = torch.stack([align_rotmat_col0, align_rotmat_col1, align_rotmat_col2], dim=1)

        targetB_align_trf = rotmat_to_trf(align_rotmat, align_translate)
        return targetB_align_trf

    def align_targetB(
        self,
        target,
        pred_local_trf,
        valid_nof
    ):
        # align targetB
        targetB = get_last_valid_block(target, valid_nof, self.block_nof)
        targetB_local_trf = to_trf(targetB, self.target_feat_start_idx, self.target_feat_end_idx, self.skel)
        if self.config.use_dest:
            # if using destination loss, make the model predict first root for targetB
            dest_root_trf = get_dest_frame(pred_local_trf, valid_nof, self.block_nof)[:, 0]
            targetB_align_dest_trf = dest_root_trf.unsqueeze(1).unsqueeze(1)
        else:
            # if not using destination loss, simply find closest trf to the last root
            last_valid_root_trf = get_last_valid_root_trf(pred_local_trf, valid_nof, self.block_nof)
            targetB_align_dest_trf = find_closest_trf_to(targetB_local_trf[:, 0, 0], last_valid_root_trf, self.device)
            targetB_align_dest_trf = targetB_align_dest_trf.unsqueeze(1).unsqueeze(1)
        targetB_trf_aligned = align(targetB_local_trf, targetB_align_dest_trf)
        return targetB_trf_aligned

'''
Mask utils
'''
def get_square_padding_mask(max_len, pred_end_idx, n_head, device=None):
    '''
    Compute mask for padding data with various lengths
    Value should be True if masking is needed

    max_len: max len for motion code sequence
    pred_end_cidx: [64]

    Returns:
        padding_mask: [batch_size, n_head, max_len, max_len]
    '''
    seq_range = torch.arange(max_len).unsqueeze(0)
    padding_mask_col = (seq_range >= pred_end_idx.unsqueeze(1)).unsqueeze(-1)
    padding_mask_row = padding_mask_col.transpose(-1, -2)

    padding_mask_col = padding_mask_col.unsqueeze(1)
    padding_mask_row = padding_mask_row.unsqueeze(1)
    # make squared mask so that rows can be ignored too

    padding_mask_col = padding_mask_col.expand(-1, n_head, -1, max_len)
    padding_mask_row = padding_mask_row.expand(-1, n_head, max_len, -1)
    padding_mask = torch.bitwise_or(padding_mask_col, padding_mask_row)
    if device is not None:
        padding_mask = padding_mask.to(device)

    return padding_mask

def get_inbetween_mask(max_len, pred_start_idx, pred_end_idx, device=None):
    '''
    Compute mask for loss computation
    Value should be True if masking is needed

    max_len: max len for motion code sequence
    pred_start_cidx: [64]
    pred_end_cidx: [64]
    '''
    seq_range = torch.arange(max_len).unsqueeze(0).to(pred_start_idx.device)

    # Mask inbetween code
    inb_mask = (seq_range >= pred_start_idx.unsqueeze(1)) & (seq_range < pred_end_idx.unsqueeze(1))

    # (batch_size, seq, seq)
    if device is not None:
        inb_mask = inb_mask.to(device)
    return inb_mask

def get_disc_mask(
    max_len: int,
    valid_nof: torch.Tensor,
    block_nof: int,
    device: torch.device
):
    half = block_nof // 2
    n_valid_block_max = int(max_len/block_nof) - 1 # two half blocks will be trimmed

    disc_valid_idx = (valid_nof - block_nof - half) // block_nof
    range_idx = torch.arange(n_valid_block_max, device=valid_nof.device).unsqueeze(0)
    disc_loss_mask = range_idx < disc_valid_idx.unsqueeze(1)

    true_pad = torch.ones_like(disc_loss_mask[:, :1], dtype=torch.bool) # one valid block will be appended at the end
    disc_loss_mask = torch.cat([disc_loss_mask, true_pad], dim=1).reshape([-1, 1])
    disc_loss_mask = disc_loss_mask.to(torch.float32).to(device)
    return disc_loss_mask