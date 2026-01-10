import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import wandb
import random
import os
from datetime import datetime
from datasets import ComprehensiveDataset
from earlyfusionresnet import EarlyFusionResNet

# UTILS
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# EXECUTION 
def save_predictions(model, loader, device, output_file):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in loader:
            *imgs, trans_log, rot_log = batch
            imgs = [img.to(device) for img in imgs]
            
            out_t_log, out_r_log = model(*imgs)
            
            # Convert Log -> Meters
            pred_t = torch.exp(out_t_log).cpu().numpy() - 1e-6
            pred_r = torch.exp(out_r_log).cpu().numpy() - 1e-6
            gt_t = (torch.exp(trans_log) - 1e-6).cpu().numpy()
            gt_r = (torch.exp(rot_log) - 1e-6).cpu().numpy()
            
            for i in range(len(pred_t)):
                results.append({
                    "pred_trans": pred_t[i], "gt_trans": gt_t[i],
                    "pred_rot": pred_r[i], "gt_rot": gt_r[i]
                })
    
    # Save CSV for Histograms
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='rgb', choices=['mean', 'rgb', 'orb', 'combined'])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--train_sequences", required=True)
    parser.add_argument("--val_sequences", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    args = parser.parse_args()
    
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.wandb:
        wandb.init(
            project="ood-slam-baselines", 
            group=f"{args.dataset}_{args.mode}",
            name=f"{args.dataset}_{args.mode}_{args.seed}", 
            config=vars(args)
        )

    def load_seq(p): 
        with open(p) as f: return [l.strip() for l in f if l.strip()]

    # Load Data
    train_ds = ComprehensiveDataset(args.data_dir, load_seq(args.train_sequences), args.dataset, args.mode, args.labels_dir)
    val_ds = ComprehensiveDataset(args.data_dir, load_seq(args.val_sequences), args.dataset, args.mode, args.labels_dir)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize Model
    if args.mode == 'rgb': model = EarlyFusionResNet(in_channels=6)
    elif args.mode == 'orb': model = EarlyFusionResNet(in_channels=6)
    elif args.mode == 'combined': model = EarlyFusionResNet(in_channels=12)
    elif args.mode == 'mean':
        # Mean baseline only
        train_trans = np.array([p['trans'] for p in train_ds.pairs])
        val_trans = np.array([p['trans'] for p in val_ds.pairs])
        val_rot = np.array([p['rot'] for p in val_ds.pairs])
        
        mean_trans = train_trans.mean()
        mean_rot = np.array([p['rot'] for p in train_ds.pairs]).mean()
        
        mse_trans = np.mean((val_trans - mean_trans) ** 2)
        mse_rot = np.mean((val_rot - mean_rot) ** 2)
        
        print(f"Mean Baseline - MSE Trans: {mse_trans:.6f}, MSE Rot: {mse_rot:.6f}")
        
        if args.wandb:
            wandb.log({"mse_trans": mse_trans, "mse_rot": mse_rot})
            wandb.finish()
        return
    
    model = model.to(device)

    # Differential Learning Rates (Stability Fix)
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': args.lr_backbone},
        {'params': model.head.parameters(), 'lr': args.lr_head}
    ], weight_decay=1e-2)
    
    criterion = nn.MSELoss()
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_dl:
            *imgs, trans, rot = batch
            imgs = [img.to(device) for img in imgs]
            trans, rot = trans.to(device), rot.to(device)
            
            optimizer.zero_grad()
            out_t, out_r = model(*imgs)
            loss = criterion(out_t, trans) + criterion(out_r, rot)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        all_trans_errs = []
        all_rot_errs = []
        with torch.no_grad():
            for batch in val_dl:
                *imgs, trans, rot = batch
                imgs = [img.to(device) for img in imgs]
                trans, rot = trans.to(device), rot.to(device)
                
                out_t, out_r = model(*imgs)
                val_loss += (criterion(out_t, trans) + criterion(out_r, rot)).item()
                
                # Metric (Meters)
                pred_m = torch.exp(out_t) - 1e-6
                gt_m = torch.exp(trans) - 1e-6
                all_trans_errs.append((pred_m - gt_m).cpu() ** 2)

                pred_r_m = torch.exp(out_r) - 1e-6
                gt_r_m = torch.exp(rot) - 1e-6
                all_rot_errs.append((pred_r_m - gt_r_m).cpu() ** 2)
        
        mse_trans = torch.cat(all_trans_errs).mean().item()
        mse_rot = torch.cat(all_rot_errs).mean().item()
        avg_val_loss = val_loss / len(val_dl)
        
        print(f"Ep {epoch+1}: Train={train_loss/len(train_dl):.4f} | Val={avg_val_loss:.4f} | MSE_Trans={mse_trans:.4f} | MSE_Rot={mse_rot:.4f}")
        
        if args.wandb:
            wandb.log({"train_loss": train_loss/len(train_dl), "val_loss": avg_val_loss, "mse_trans": mse_trans, "mse_rot": mse_rot})
            
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs("/scratch/adamb14/ood_slam/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"/scratch/adamb14/ood_slam/checkpoints/best_model_{args.dataset}_{args.mode}_seed{args.seed}.pth")

    # Final Export
    print("Exporting predictions...")
    model.load_state_dict(torch.load(f"/scratch/adamb14/ood_slam/checkpoints/best_model_{args.dataset}_{args.mode}_seed{args.seed}.pth"))
    save_predictions(model, val_dl, device, f"preds_{args.dataset}_{args.mode}_seed{args.seed}.csv")

if __name__ == "__main__":
    main()