import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np

# DATASET 
class ComprehensiveDataset(Dataset):
    def __init__(self, data_dir, sequences, dataset_type, mode, labels_dir=None, img_size=(224, 224)):
        self.data_dir = Path(data_dir)
        self.labels_dir = Path(labels_dir) if labels_dir else self.data_dir
        self.mode = mode
        self.dataset_type = dataset_type
        self.pairs = []
        
        # TRANSFORMS
        self.rgb_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # ORB: Sparse input, just tensor
        self.orb_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

        print(f"Indexing {dataset_type} sequences for mode '{mode}'...")
        for seq in sequences:
            self._load_sequence(seq)
        print(f"Loaded {len(self.pairs)} pairs.")

    def _get_paths(self, seq, seq_dir):
        # 1. RGB Path
        if self.dataset_type == "kitti":
            rgb_dir = seq_dir / "image_0"
        elif self.dataset_type == "euroc":
            rgb_dir = seq_dir / "mav0" / "cam0" / "data"
        elif self.dataset_type == "tartanair":
            rgb_dir = seq_dir / "image_left"
        elif self.dataset_type == "eth3d":
            rgb_dir = seq_dir / "rgb"
        else:
            raise ValueError("Unknown Dataset")

        # 2. ORB Path (Standardized)
        orb_dir = seq_dir / "orb_semantic"
        
        # 3. Label Path
        label_path = self.labels_dir / seq / "labels.csv"
        
        return rgb_dir, orb_dir, label_path

    def _load_sequence(self, seq):
        seq_dir = self.data_dir / seq
        rgb_dir, orb_dir, label_path = self._get_paths(seq, seq_dir)
        
        if not label_path.exists(): return
        if self.mode in ['rgb', 'combined'] and not rgb_dir.exists(): return
        if self.mode in ['orb', 'combined'] and not orb_dir.exists(): return

        df = pd.read_csv(label_path)
        
        # Build index -> filename mapping from sorted directory listing
        all_files = sorted(rgb_dir.glob("*.png"))
        idx_to_fname = {i: f.name for i, f in enumerate(all_files)}

        for i, row in df.iterrows():
            if row.get('exists', 1) != 1 or pd.isna(row['rpe_trans']): continue
            if i + 1 >= len(df): break
            
            curr_id = int(row.iloc[0])
            next_id = int(df.iloc[i+1, 0])
            
            if curr_id not in idx_to_fname or next_id not in idx_to_fname:
                continue
            
            curr_fname = idx_to_fname[curr_id]
            next_fname = idx_to_fname[next_id]
                
            item = {
                'trans': np.float32(row['rpe_trans']),
                'rot': np.float32(row['rpe_rot'])
            }
            
            if self.mode in ['rgb', 'combined']:
                item['rgb1'] = str(rgb_dir / curr_fname)
                item['rgb2'] = str(rgb_dir / next_fname)
            
            if self.mode in ['orb', 'combined']:
                item['orb1'] = str(orb_dir / curr_fname)
                item['orb2'] = str(orb_dir / next_fname)
            
            self.pairs.append(item)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        
        # Log Targets
        trans_log = np.log(item['trans'] + 1e-6)
        rot_log = np.log(item['rot'] + 1e-6)
        target = (torch.tensor(trans_log), torch.tensor(rot_log))

        inputs = []
        if self.mode in ['rgb', 'combined']:
            rgb1 = Image.open(item['rgb1']).convert('RGB')
            rgb2 = Image.open(item['rgb2']).convert('RGB')
            inputs.extend([self.rgb_transform(rgb1), self.rgb_transform(rgb2)])

        if self.mode in ['orb', 'combined']:
            orb1 = Image.open(item['orb1']).convert('RGB')
            orb2 = Image.open(item['orb2']).convert('RGB')
            inputs.extend([self.orb_transform(orb1), self.orb_transform(orb2)])

        return *inputs, target[0], target[1]