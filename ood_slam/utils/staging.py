import os 
import subprocess
import logging

log = logging.getLogger(__name__)

def stage_dataset(cfg):
    """
    Stages data to SLURM_TMPDIR based on the config.
    """
    dataset_type = cfg.data.dataset_type

    # Get the destination
    dest = os.getenv('SLURM_TMPDIR', None)
    if dest is None:
        log.warning("SLURM_TMPDIR not set; skipping staging.")
        return
    
    log.info(f"Staging dataset '{dataset_type}' to {dest}...")

    repo_root = os.getcwd()

    # kitti staging logic
    if dataset_type == 'kitti':
        data_source = "/home/adamb14/scratch/ood_slam/datasets/kitti/raw/sequences"
        results_source = "/home/adamb14/scratch/ood_slam/results/kitti/orbslam2"
        
        # Load sequence lists
        # We assume the split files are in repo_root/splits
        splits = ["splits/kitti_train.txt", "splits/kitti_val.txt"]
        seqs = []
        for s_file in splits:
            path = os.path.join(repo_root, s_file)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    seqs.extend([line.strip() for line in f if line.strip()])
        
        for seq in seqs:
            seq_dir = os.path.join(dest, seq)
            os.makedirs(seq_dir, exist_ok=True)
            
            # Extract Tar
            archive = os.path.join(data_source, f"{seq}.tar.gz")
            if os.path.exists(archive):
                log.info(f"Extracting {seq}...")
                subprocess.run(["tar", "-xf", archive, "-C", seq_dir], check=True)
            
            # Copy Labels
            label_src = os.path.join(results_source, seq, "labels.csv")
            if os.path.exists(label_src):
                subprocess.run(["cp", label_src, seq_dir], check=True)
            else:
                log.warning(f"No labels found for {seq}")

    # tartanair
    elif dataset_type == "tartanair":
        data_source = "/home/adamb14/scratch/ood_slam/datasets/tartanair_zipped"
        results_source = "/home/adamb14/scratch/ood_slam/results/tartanair/orbslam2"
        
        splits = ["splits/tartanair_train.txt", "splits/tartanair_val.txt"]
        seqs = []
        for s_file in splits:
            path = os.path.join(repo_root, s_file)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    seqs.extend([line.strip() for line in f if line.strip()])

        for seq in seqs:
            # seq: "abandonedfactory/Easy/P001"
            parent_dir = os.path.join(dest, os.path.dirname(seq))
            os.makedirs(parent_dir, exist_ok=True)
            
            # Unzip
            zip_file = os.path.join(data_source, f"{seq}.zip")
            if os.path.exists(zip_file):
                # -q: quiet, -o: overwrite, -d: dest
                subprocess.run(["unzip", "-q", "-o", zip_file, "-d", parent_dir], check=True)
            
            # Copy Labels
            seq_dest_dir = os.path.join(dest, seq)
            os.makedirs(seq_dest_dir, exist_ok=True)
            label_src = os.path.join(results_source, seq, "labels.csv")
            if os.path.exists(label_src):
                subprocess.run(["cp", label_src, seq_dest_dir], check=True)
            else:
                log.warning(f"No labels found for {seq}")
    
    elif dataset_type == "euroc":
        data_source = "/home/adamb14/scratch/ood_slam/datasets/euroc"
        results_source = "/home/adamb14/scratch/ood_slam/results/euroc/orbslam2"
        
        splits = ["splits/euroc_train.txt", "splits/euroc_val.txt"]
        seqs = []
        for s_file in splits:
            path = os.path.join(repo_root, s_file)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    seqs.extend([line.strip() for line in f if line.strip()])

        for seq in seqs:
            # seq: "V1_01_easy"
            seq_dir = os.path.join(dest, seq)
            os.makedirs(seq_dir, exist_ok=True)
            
            # Unzip
            zip_file = os.path.join(data_source, f"{seq}", f"{seq}.zip")
            if os.path.exists(zip_file):
                # -q: quiet, -o: overwrite, -d: dest
                subprocess.run(["unzip", "-q", "-o", zip_file, "-d", seq_dir], check=True)
            
            # Copy Labels
            label_src = os.path.join(results_source, seq, "labels.csv")
            if os.path.exists(label_src):
                subprocess.run(["cp", label_src, seq_dir], check=True)
            else:
                log.warning(f"No labels found for {seq}")
                
    else:
        log.error(f"Unknown dataset_type for staging: {dataset_type}")
        
    log.info("STAGING DONE")