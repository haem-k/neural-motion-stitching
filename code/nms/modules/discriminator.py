import torch
import torch.nn as nn
import torch.nn.functional as F
import nms.utils.utils as utils

class Discriminator(nn.Module):
    def __init__(self, config, total_joints, stats_path):
        super(Discriminator, self).__init__()
        
        # init jnt_map and input dim
        target_joints = config.joint_list
        jnt_map = utils.get_joint_map(total_joints, target_joints)
        jnt_map = torch.from_numpy(jnt_map).to(torch.int32)
        disc_input_dim = jnt_map.shape[0] * 3 # disc with selected joints
        self.register_buffer("jnt_map", jnt_map)
        
        # load statistics
        disc_statistics = torch.load(stats_path)
        for key, stat in disc_statistics.items():
            stat = stat.reshape([-1, 3])
            stat = stat[self.jnt_map]
            self.register_buffer(key, stat.flatten())
        
        # setup var for network layers
        self.input_dim = disc_input_dim
        self.input_len = config.block_nof
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.head
        self.w_gp = config.w_gp
        
        self._conv = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim // 2, self.hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.linear_dim = (self.input_len // 4) * self.hidden_dim
        self._linear = nn.Sequential(
            nn.Linear(self.linear_dim, self.linear_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0),
            nn.Linear(self.linear_dim // 2, self.output_dim),
        )
        
    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        x_conv = x_norm.permute(0, 2, 1)
        x_conv = self._conv(x_conv)
        x_conv = x_conv.permute(0, 2, 1)
        
        batch_size = x_conv.shape[0]
        x_conv = x_conv.reshape([batch_size, -1])
        x_out = self._linear(x_conv)
        x_out = torch.mean(x_out, dim=-1, keepdim=True)
        return x_out, x_norm

    def compute_loss(self, fake_logit, real_logit, mask, real_input=None):
        fake_label = torch.zeros_like(fake_logit).to(fake_logit.device)
        real_label = torch.ones_like(real_logit).to(real_logit.device)
        
        loss_fake = 0.5 * F.mse_loss(fake_logit, fake_label, reduction="none")
        loss_real = 0.5 * F.mse_loss(real_logit, real_label, reduction="none")
        
        loss_fake = (loss_fake * mask).sum() / mask.sum()
        loss_real = (loss_real * mask).sum() / mask.sum()
        
        real_grad_penalty = torch.zeros((1)).to(fake_logit.device)
        if real_input is not None:
            # R1 regularization
            real_grad = torch.autograd.grad(real_logit, real_input, grad_outputs=torch.ones_like(real_logit), create_graph=True, retain_graph=True)
            real_grad = real_grad[0]
            real_grad = torch.sum(torch.square(real_grad), dim=-1)
            real_grad_penalty = torch.mean(real_grad)
        
        loss = loss_real + loss_fake + self.w_gp * real_grad_penalty
        loss_dict = {
            "loss_disc": loss,
            "loss_real": loss_real,
            "loss_fake": loss_fake,
            "grad_penalty": real_grad_penalty
        }
        return loss_dict
