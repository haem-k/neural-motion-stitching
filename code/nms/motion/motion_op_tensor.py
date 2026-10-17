import torch

def fk(local_rots: torch.Tensor, offsets: torch.Tensor, parent_indices: list):
    # Need to return the full transforms in world frame
    # FK
    '''
    Returns transformation of joints in world frame [nom, nof, noj, 4, 4]

    Args:
        local_rot: torch tensor with parent relative joint rotations [nom, nof, noj, 4, 4]
                   Note that for the first joint(root), it is world frame transformation.
                   Should not contain parent relative translation values. Will be computed with offsets
        skel: Skeleton of a given character for FK order
    '''
    nom, nof, noj = local_rots.shape[:3]

    # Initialize world transformation tensor
    world_trf_list = [local_rots[..., 0, :, :]]

    # ! Assumes that joint order is ordered this way: parent --> child
    for i in range(1, noj):
        parent_idx = parent_indices[i]
        parent_world = world_trf_list[parent_idx]
        parent_rel_trf = torch.matmul(offsets[i], local_rots[..., i, :, :])

        # Compute world transformation for current joint
        trf = torch.matmul(parent_world, parent_rel_trf)
        world_trf_list.append(trf)
    world_trf = torch.stack(world_trf_list, dim=-3)
    return world_trf
