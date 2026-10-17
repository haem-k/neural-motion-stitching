import numpy as np
import torch
import torch.nn.functional as F
import nms.utils.utils as utils
from torch.utils.data import Dataset

class VQVAEDataset(Dataset):
    def __init__(self, data_dir) -> None:
        self.data = torch.load(data_dir)
        self.total_seq_num = self.data.shape[0]

    def get_data_shape(self):
        return self.data.shape

    def get_mean_std(self):
        # data [batch, nof, dim]
        reshaped_data = self.data.reshape((-1, self.data.shape[-1]))
        mean = reshaped_data.mean(0)
        std = reshaped_data.std(0) + 1e-7 # prevent division by zero
        return mean, std

    def __getitem__(self, index) -> torch.Tensor:
        sample = self.data[index]
        return index, sample

    def __len__(self) -> int:
        return self.total_seq_num

class MGDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        max_len: int,
        block_nof: int,
        mode: str = "train",
        lengths: list = None,
        use_target: bool = True,
    ):
        self.data_dir = data_dir
        self.max_len = max_len
        self.block_nof = block_nof
        self.mode = mode
        self.group = ["gt", "guide", "target"]
        self.use_target = use_target

        # get data file names and number of samples per file
        if lengths is not None:
            self.collect_data_fnames_count_lengths(lengths)
        else:
            self.collect_data_fnames_count()
        self.collect_datapair_dim()

    def collect_data_fnames_count(self):

        self.dir_dict = {}
        for i in self.group:
            self.dir_dict[i] = f"{self.data_dir}/{i}"

        self.n_samples = []
        self.fname_dict = {}
        for i in self.group:
            self.fname_dict[i] = utils.list_files_in_dir(self.dir_dict[i], numerical=False)

        assert len(self.fname_dict["gt"]) == len(self.fname_dict["guide"]) and len(self.fname_dict["guide"]) == len(self.fname_dict["target"]), "Data pairs needs to be matched"

        # Get number of samples
        gt_dir = self.dir_dict["gt"]
        for i in self.fname_dict["gt"]:
            if not utils.is_pt(i):
                continue
            data = torch.load(f"{gt_dir}/{i}")
            self.n_samples.append(data.shape[0])
        self.nsamples_cumsum = np.cumsum(self.n_samples)

    def collect_data_fnames_count_lengths(self, lengths):
        self.dir_dict = {}
        for i in self.group:
            self.dir_dict[i] = f"{self.data_dir}/{i}"

        self.n_samples = []
        self.fname_dict = {}
        for i in self.group:
            # self.fname_dict[i] = utils.list_files_in_dir(self.dir_dict[i], numerical=False)
            all_fnames = utils.list_files_in_dir(self.dir_dict[i], numerical=False)
            prefixes = tuple(f"{i}_{j}" for j in lengths)
            self.fname_dict[i] = [f for f in all_fnames if f.startswith(prefixes)]

        assert len(self.fname_dict["gt"]) == len(self.fname_dict["guide"]) and len(self.fname_dict["guide"]) == len(self.fname_dict["target"]), "Data pairs needs to be matched"
        # Get number of samples for each file pair (only count GT)
        # n_files = len(self.gt_fnames)
        g = "target"
        for i in self.fname_dict[g]:
            if not utils.is_pt(i):
                continue
            data = torch.load(f"{self.dir_dict[g]}/{i}")
            self.n_samples.append(data.shape[0])
        self.nsamples_cumsum = np.cumsum(self.n_samples)


    def collect_datapair_dim(self):
        # save GT, input dim
        gt_dir = self.dir_dict["gt"]
        gt_fname = self.fname_dict["gt"][0]
        gt_file = f"{gt_dir}/{gt_fname}"
        gt_data = torch.load(gt_file)
        self.gt_dim = gt_data.shape[-1]

        target_dir = self.dir_dict["target"]
        target_fname = self.fname_dict["target"][0]
        target_file = f"{target_dir}/{target_fname}"
        target_data = torch.load(target_file)
        target_dim = target_data.shape[-1]

        guide_dir = self.dir_dict["guide"]
        guide_fname = self.fname_dict["guide"][0]
        guide_file = f"{guide_dir}/{guide_fname}"
        guide_data = torch.load(guide_file)
        guide_dim = guide_data.shape[-1]
        self.input_dim = target_dim + guide_dim

    def __getitem__(self, index):
        # find motion sample of given index
        length_set_idx = np.searchsorted(self.nsamples_cumsum, index, side='left') # find motion to get data from
        length_set_idx = max(0, length_set_idx) # ensure index is non-negative

        prev_samples_sum = 0
        if length_set_idx != 0:
            prev_samples_sum = self.nsamples_cumsum[length_set_idx - 1]

        residual_samples = index - prev_samples_sum # number of sample within a single motion
        sample_idx = max(0, residual_samples - 1)

        sample_dict = {}
        for i in self.group:
            file = torch.load(f"{self.dir_dict[i]}/{self.fname_dict[i][length_set_idx]}")
            sample_dict[i] = file[sample_idx]

        gt_sample = sample_dict["gt"]
        target_sample = sample_dict["target"]
        guide_sample = sample_dict["guide"]
        valid_nof = gt_sample.shape[0]

        # fill target
        inbetween_pad_size = valid_nof - self.block_nof * 2
        zero = torch.zeros([inbetween_pad_size, target_sample.shape[-1]])
        target = torch.cat([target_sample[:self.block_nof], zero, target_sample[-self.block_nof:]])


        # make input
        input_sample = torch.cat([target, guide_sample], dim=-1)
        if not self.use_target: # if use target is false, use only guide
            input_sample = guide_sample

        # pad to match length
        pad_size = self.max_len - valid_nof
        pad = (0, 0, 0, pad_size) # left, right, top, bottom
        gt_padded = F.pad(gt_sample, pad)
        input_padded = F.pad(input_sample, pad)
        return index, valid_nof, gt_padded, input_padded

    def __len__(self) -> int:
        return self.nsamples_cumsum[-1]