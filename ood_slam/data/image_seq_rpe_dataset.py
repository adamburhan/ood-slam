import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import time
from ood_slam.data.utils import normalize_angle_delta


def get_data_info(folder_list, seq_len_range, overlap, sample_times=1, 
                  data_dir=None, error_dir=None, image_dir=None, pad_y=False, shuffle=False, sort=True):
    X_path, Y_trans, Y_rot = [], [], []
    X_len = []

    assert seq_len_range[0] == seq_len_range[1], "Only fixed sequence lengths are supported for now."
    seq_len = seq_len_range[0]

    for folder in folder_list:
        start_t = time.time()

        # Load error magnitudes
        errors_df = pd.read_csv(f'{error_dir}/{folder}_rpe_labels.csv')
        errors_trans = errors_df['rpe_translation'].to_numpy()  # (n_images-1, )
        errors_rot = errors_df['rpe_rotation'].to_numpy()  # (n_images-1, )

        # Load image paths
        fpaths = glob.glob('{}{}/*.png'.format(image_dir, folder))
        fpaths.sort()
        n_images = len(fpaths)
        
        # sanity check
        if len(errors_trans) == n_images - 1:
            # Pad a dummy at the beggining so that:
            # error[t] = RPE between frame t-1 and t, t >= 1
            # error[0] = -1 (invalid value)
            errors_trans = np.insert(errors_trans, 0, -1)
            errors_rot = np.insert(errors_rot, 0, -1)
        elif len(errors_trans) != n_images:
            raise ValueError(
                f"Unexpected length mismatch in folder {folder}: "
                f"{len(errors_trans)=} vs {n_images=} images"
            )
        
        if sample_times > 1:
            sample_interval = int(np.ceil(seq_len_range[0] / sample_times))
            start_frames = list(range(0, seq_len_range[0], sample_interval))
            print('Sample start from frame {}'.format(start_frames))
        else:
            start_frames = [0]

        for st in start_frames:
            n_frames = len(fpaths) - st
            jump = seq_len - overlap
            res = n_frames % seq_len
            if res != 0:
                n_frames = n_frames - res

            for i in range(st, st + n_frames, jump):
                x_seg = fpaths[i : i + seq_len]
                trans_seg = errors_trans[i : i + seq_len]
                rot_seg = errors_rot[i : i + seq_len]

                # sanity check lengths
                if len(x_seg) != seq_len or len(trans_seg) != seq_len:
                    continue

                # We will use indices 1...seq_len-1 for the loss (because of y[:,1:])
                # so require those to be valid (not -1)
                if np.any(trans_seg[1:] < 0) or np.any(rot_seg[1:] < 0):
                    continue
            
                X_path.append(x_seg)
                Y_trans.append(trans_seg)
                Y_rot.append(rot_seg)
                X_len.append(len(x_seg))
        print('Folder {} finish in {} sec'.format(folder, time.time()-start_t))
    
    # Convert to pandas dataframes
    data = {
        'seq_len': X_len, 
        'image_path': X_path, 
        'rpe_translation': Y_trans, 
        'rpe_rotation': Y_rot
    }
    df = pd.DataFrame(data, columns = ['seq_len', 'image_path', 'rpe_translation', 'rpe_rotation'])
    # Shuffle through all videos
    if shuffle:
        df = df.sample(frac=1)
    # Sort dataframe by seq_len
    if sort:
        df = df.sort_values(by=['seq_len'], ascending=False)
    return df

class SortedRandomBatchSampler(Sampler):
    def __init__(self, info_dataframe, batch_size, drop_last=False):
        self.df = info_dataframe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.unique_seq_lens = sorted(self.df.iloc[:].seq_len.unique(), reverse=True)
        # Calculate len (num of batches, not num of samples)
        self.len = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            self.len += n_batch

    def __iter__(self):
        
        # Calculate number of sameples in each group (grouped by seq_len)
        list_batch_indexes = []
        start_idx = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            rand_idxs = (start_idx + torch.randperm(n_sample)).tolist()
            tmp = [rand_idxs[s*self.batch_size: s*self.batch_size+self.batch_size] for s in range(0, n_batch)]
            list_batch_indexes += tmp
            start_idx += n_sample
        return iter(list_batch_indexes)

    def __len__(self):
        return self.len
    
class ImageSeqErrorRegDataset(Dataset):
    def __init__(
        self,
        info_dataframe,
        resize_mode="crop",
        new_size=None,
        img_mean=None,
        img_std=(1, 1, 1),
        minus_point_5=False
    ):
        # Transforms
        transform_ops = []
        if resize_mode == "crop":
            transform_ops.append(transforms.CenterCrop((new_size[0], new_size[1])))
        elif resize_mode == "rescale":
            transform_ops.append(transforms.Resize((new_size[0], new_size[1])))
        transform_ops.append(transforms.ToTensor())
        
        self.transformer = transforms.Compose(transform_ops)
        self.minus_point_5 = minus_point_5
        self.normalizer = transforms.Normalize(mean=img_mean, std=img_std)
        
        self.data_info = info_dataframe
        self.seq_len_list = list(self.data_info.seq_len)
        self.image_arr = np.asarray(self.data_info.image_path)
        self.trans_arr = np.asarray(self.data_info.rpe_translation)
        self.rot_arr = np.asarray(self.data_info.rpe_rotation)
        
    def __getitem__(self, index):
        # load images
        image_path_sequence = self.image_arr[index]
        sequence_len = torch.tensor(self.seq_len_list[index])  #sequence_len = torch.tensor(len(image_path_sequence))
        
        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_tensor = self.transformer(img_as_img)
            if self.minus_point_5:
                img_as_tensor = img_as_tensor - 0.5  # from [0, 1] -> [-0.5, 0.5]
            img_as_tensor = self.normalizer(img_as_tensor)
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)

        # load rpes
        trans_seq = self.trans_arr[index]
        rot_seq = self.rot_arr[index]

        # Make them (T, 1) float tensors
        trans_seq = torch.as_tensor(trans_seq, dtype=torch.float32).unsqueeze(-1)
        rot_seq   = torch.as_tensor(rot_seq, dtype=torch.float32).unsqueeze(-1)

        return (sequence_len, image_sequence, trans_seq, rot_seq)

    def __len__(self):
        return len(self.data_info.index)
            
    