"""Base model interface for ood-slam models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Base class for all ood-slam models."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass
    
    @abstractmethod
    def get_loss(self, batch: Tuple[Any, ...]) -> torch.Tensor:
        """Compute loss for a batch."""
        pass
    
    def training_step(self, batch: Tuple[Any, ...], optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Execute one training step."""
        self.train()
        optimizer.zero_grad()
        
        loss = self.get_loss(batch)
        loss.backward()
        
        # Gradient clipping if specified
        if hasattr(self, 'clip_grad_norm') and self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
            
        optimizer.step()
        
        return {"loss": loss.item()}
    
    @torch.no_grad()
    def validation_step(self, batch: Tuple[Any, ...]) -> Dict[str, float]:
        """Execute one validation step."""
        self.eval()
        loss = self.get_loss(batch)
        return {"val_loss": loss.item()}