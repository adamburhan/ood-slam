import torch
from torch import Tensor
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, Any, Optional
import logging

from tqdm import tqdm
import os

logger = logging.getLogger(__name__)


import wandb


@torch.no_grad()
def eval_model(model, loader, device):
    """Evaluate model on validation set."""
    model.eval()
    metrics = defaultdict(list)
    
    for batch in tqdm(loader, desc="Validation", leave=False):
        # Move batch to device
        batch = [x.to(device) if torch.is_tensor(x) else x for x in batch]
        val_metrics = model.validation_step(batch)
        
        for key, value in val_metrics.items():
            metrics[key].append(value)
    
    # Average metrics
    avg_metrics = {key: sum(values) / len(values) for key, values in metrics.items()}
    return avg_metrics


def train(model, train_dl, val_dl, device, config):
    """
    Training loop with optional wandb logging
    
    Args:
        model: The model to train (should inherit from BaseModel)
        train_dl: Training data loader
        val_dl: Validation data loader 
        device: Device to train on
        config: Training configuration
    """
    # Setup
    epochs = config.max_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    model.to(device)
    
    use_wandb = getattr(config, 'use_wandb', False)
    if use_wandb:
        logger.info("Wandb logging enabled")
    
    # Metrics tracking
    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)
    
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(epochs), desc="Training"):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        epoch_train_metrics = defaultdict(list)
        
        for batch in tqdm(train_dl, desc="Training", leave=False):
            # Move batch to device
            batch = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            
            # Training step
            batch_metrics = model.training_step(batch, optimizer)
            
            # Accumulate metrics
            for key, value in batch_metrics.items():
                epoch_train_metrics[key].append(value)
        
        # Average training metrics
        avg_train_metrics = {
            f"train_{key}": sum(values) / len(values) 
            for key, values in epoch_train_metrics.items()
        }
        
        # Validation phase
        if val_dl is not None:
            avg_val_metrics = eval_model(model, val_dl, device)
        else:
            avg_val_metrics = {}
        
        # Log metrics
        all_metrics = {**avg_train_metrics, **avg_val_metrics}
        all_metrics["epoch"] = epoch + 1
        logger.info(f"Epoch {epoch+1} metrics: {all_metrics}")
        
        # Log to wandb if enabled
        if use_wandb:
            wandb.log(all_metrics)
        
        # Save metrics
        for key, value in avg_train_metrics.items():
            train_metrics[key].append(value)
        for key, value in avg_val_metrics.items():
            val_metrics[key].append(value)
        
        # Save best model
        current_val_loss = avg_val_metrics.get('val_loss', float('inf'))
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            try:
                checkpoint_dir = config.get('checkpoint_dir', '.')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"New best validation loss: {best_val_loss:.4f} - saved to {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
                # Continue training even if checkpoint fails
    
    return {"train": dict(train_metrics), "val": dict(val_metrics)}
            