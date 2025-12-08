import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
from ood_slam.data.image_seq_rpe_dataset import SortedRandomBatchSampler, ImageSequenceErrorDataset, get_data_info


class ImageSequenceErrorDataModule:
    def __init__(
        self,
        data_dir: str,
        train_sequences: list,
        valid_sequences: list,
        img_means: tuple,
        img_stds: tuple,
        num_workers: int = 4,
        pin_memory: bool = True,
        sample_times: int = 3,
        img_w: int = 608,
        img_h: int = 184,
        seq_len: tuple = (5, 5),
        batch_size: int = 8,
        resize_mode: str = "rescale",
        minus_point_5: bool = True,
        use_cache: bool = True,
        cache_dir: str = None,
        overfit: bool = False,
        task: str = "regression",
    ):
        self.data_dir = data_dir
        self.train_sequences = train_sequences
        self.valid_sequences = valid_sequences
        self.img_means = img_means
        self.img_stds = img_stds
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sample_times = sample_times
        self.img_w = img_w
        self.img_h = img_h
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.resize_mode = resize_mode
        self.minus_point_5 = minus_point_5
        self.use_cache = use_cache
        self.overfit = overfit
        self.task = task
        
        # Set up cache directory like original
        if cache_dir is None:
            self.cache_dir = os.path.join(data_dir, "datainfo")
        else:
            self.cache_dir = cache_dir
            
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Generate cache file paths like original
        suffix = f't{"".join(self.train_sequences)}_v{"".join(self.valid_sequences)}_' \
         f'seq{self.seq_len[0]}x{self.seq_len[1]}_sample{self.sample_times}_task{self.task}.pickle'

        self.train_cache_path = os.path.join(self.cache_dir, f'train_df_{suffix}')
        self.valid_cache_path = os.path.join(self.cache_dir, f'valid_df_{suffix}')
    
    def setup(self):
        """Set up datasets - loads or creates data info like original DeepVO."""
        
        # Check for cached data like original implementation
        if (self.use_cache and 
            os.path.isfile(self.train_cache_path) and 
            os.path.isfile(self.valid_cache_path)):
            print(f'Load data info from {self.train_cache_path}')
            self.train_df = pd.read_pickle(self.train_cache_path)
            self.valid_df = pd.read_pickle(self.valid_cache_path)
        else:
            print('Create new data info')
            # Create train dataset info
            self.train_df = get_data_info(
                folder_list=self.train_sequences,
                seq_len_range=self.seq_len,
                overlap=1,
                sample_times=self.sample_times,
                data_dir=self.data_dir,
                error_dir=f"{self.data_dir}/errors/",  
                image_dir=f"{self.data_dir}/images/",   
                label_mode=self.task,
                sort=True
            )
            
            # Create validation dataset info
            self.valid_df = get_data_info(
                folder_list=self.valid_sequences,
                seq_len_range=self.seq_len,
                overlap=1,
                sample_times=self.sample_times,
                data_dir=self.data_dir,
                error_dir=f"{self.data_dir}/errors/",
                image_dir=f"{self.data_dir}/images/",
                label_mode=self.task,
                sort=True
            )
            
            # Save cache like original
            if self.use_cache:
                self.train_df.to_pickle(self.train_cache_path)
                self.valid_df.to_pickle(self.valid_cache_path)
        
        if self.overfit:
            # Reduce to just one batch for overfitting
            self.train_df = self.train_df.iloc[:self.batch_size]
            self.valid_df = self.train_df.copy()
            print('Overfitting mode: using only one batch of data')

        # Create datasets
        self.train_dataset = ImageSequenceErrorDataset(
            self.train_df, 
            resize_mode=self.resize_mode,
            new_size=(self.img_w, self.img_h),  
            img_mean=self.img_means, 
            img_std=self.img_stds, 
            minus_point_5=self.minus_point_5
        )
        
        self.valid_dataset = ImageSequenceErrorDataset(
            self.valid_df, 
            resize_mode=self.resize_mode,
            new_size=(self.img_w, self.img_h),
            img_mean=self.img_means, 
            img_std=self.img_stds, 
            minus_point_5=self.minus_point_5
        )
        
        print('Number of samples in training dataset: ', len(self.train_df.index))
        print('Number of samples in validation dataset: ', len(self.valid_df.index))
        print('='*50)
        
    def train_dataloader(self):
        """Create training dataloader with custom sampler like original."""
        sampler = SortedRandomBatchSampler(
            self.train_df, 
            batch_size=self.batch_size, 
            drop_last=True
        )
        return DataLoader(
            self.train_dataset, 
            batch_sampler=sampler, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self):
        """Create validation dataloader with custom sampler like original."""
        sampler = SortedRandomBatchSampler(
            self.valid_df, 
            batch_size=self.batch_size, 
            drop_last=True
        )
        return DataLoader(
            self.valid_dataset, 
            batch_sampler=sampler, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )