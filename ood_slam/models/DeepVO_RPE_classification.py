# from https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/master/model.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from typing import Tuple, Any, Dict
from .base_model import BaseModel

def conv(batch_norm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )        
        
class DeepVOErrorClassification(BaseModel):
    def __init__(
        self, 
        img_h, 
        img_w, 
        clip, 
        conv_dropout,
        rnn_hidden_size,
        rnn_dropout_between,
        rnn_dropout_out,
        num_classes,
        pretrained_flownet=None,
        pretrained_deepvo=None,
        batch_norm=True):
        super().__init__()
        # CNN
        self.batch_norm = batch_norm
        self.clip_grad_norm = clip  # Renamed for BaseModel compatibility
        self.pretrained_flownet = pretrained_flownet
        self.pretrained_deepvo = pretrained_deepvo
        
        self.conv1   = conv(self.batch_norm,   6,   64, kernel_size=7, stride=2, dropout=conv_dropout[0])
        self.conv2   = conv(self.batch_norm,  64,  128, kernel_size=5, stride=2, dropout=conv_dropout[1])
        self.conv3   = conv(self.batch_norm, 128,  256, kernel_size=5, stride=2, dropout=conv_dropout[2])
        self.conv3_1 = conv(self.batch_norm, 256,  256, kernel_size=3, stride=1, dropout=conv_dropout[3])
        self.conv4   = conv(self.batch_norm, 256,  512, kernel_size=3, stride=2, dropout=conv_dropout[4])
        self.conv4_1 = conv(self.batch_norm, 512,  512, kernel_size=3, stride=1, dropout=conv_dropout[5])
        self.conv5   = conv(self.batch_norm, 512,  512, kernel_size=3, stride=2, dropout=conv_dropout[6])
        self.conv5_1 = conv(self.batch_norm, 512,  512, kernel_size=3, stride=1, dropout=conv_dropout[7])
        self.conv6   = conv(self.batch_norm, 512, 1024, kernel_size=3, stride=2, dropout=conv_dropout[8])
        # Get the shape based on different image sizes
        with torch.no_grad():
            tmp = torch.zeros(1, 6, img_h, img_w, device=self.conv1[0].weight.device)
            tmp = self.encode_image(tmp)
            cnn_dim = tmp.view(1, -1).size(1)
        
        # RNN
        self.rnn = nn.LSTM(
            input_size=cnn_dim,
            hidden_size=rnn_hidden_size,
            num_layers=2,
            dropout=rnn_dropout_between,
            batch_first=True
        )
        self.rnn_drop_out = nn.Dropout(rnn_dropout_out)
        # Two classification heads for translation and rotation error classes
        self.linear_trans = nn.Linear(in_features=rnn_hidden_size, out_features=num_classes)
        self.linear_rot = nn.Linear(in_features=rnn_hidden_size, out_features=num_classes)
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                # layer 1
                kaiming_normal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n//4, n//2
                m.bias_hh_l0.data[start:end].fill_(1.)
                
                # layer 2
                kaiming_normal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n//4, n//2
                m.bias_hh_l1.data[start:end].fill_(1.)
                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        # Load pretrained FlowNet weights 
        if self.pretrained_flownet is not None and self.pretrained_deepvo is None:
            self._load_pretrained_flownet(self.pretrained_flownet)
        # Load pretrained DeepVO weights
        if self.pretrained_deepvo is not None:
            self._load_pretrained_deepvo(self.pretrained_deepvo)
    
    def _load_pretrained_flownet(self, pretrained_path: str):
        """
        Load FlowNet pretrained weights for the CNN layers.
        
        Args:
            pretrained_path: Path to the pretrained FlowNet model
        """
        import os
        if not os.path.exists(pretrained_path):
            print(f"Warning: Pretrained FlowNet model not found at {pretrained_path}")
            return
            
        try:
            # Load pretrained weights
            device = next(self.parameters()).device
            if device.type == 'cuda':
                pretrained_w = torch.load(pretrained_path)
            else:
                pretrained_w = torch.load(pretrained_path, map_location='cpu')
            
            print(f"Loading FlowNet pretrained model from {pretrained_path}")
            
            # Use only conv-layer-part of FlowNet as CNN for DeepVO
            model_dict = self.state_dict()
            
            # Filter pretrained weights to match DeepVO conv layers
            if 'state_dict' in pretrained_w:
                pretrained_dict = pretrained_w['state_dict']
            else:
                pretrained_dict = pretrained_w
                
            # Only update conv layers that exist in both models
            update_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and 'conv' in k:
                    # Check if shapes match
                    if v.shape == model_dict[k].shape:
                        update_dict[k] = v
                        print(f"  Loading: {k} {v.shape}")
                    else:
                        print(f"  Skipping {k}: shape mismatch {v.shape} vs {model_dict[k].shape}")
            
            if update_dict:
                model_dict.update(update_dict)
                self.load_state_dict(model_dict)
                print(f"Successfully loaded {len(update_dict)} conv layers from pretrained FlowNet")
            else:
                print("Warning: No matching conv layers found in pretrained model")
                
        except Exception as e:
            print(f"Error loading pretrained FlowNet weights: {e}")

    def _load_pretrained_deepvo(self, pretrained_path: str):
        """
        Load pretrained DeepVO weights for the entire model.
        
        Args:
            pretrained_path: Path to the pretrained DeepVO model
        """
        import os
        if not os.path.exists(pretrained_path):
            print(f"Warning: Pretrained DeepVO model not found at {pretrained_path}")
            return
            
        try:
            # Load pretrained weights
            device = next(self.parameters()).device
            if device.type == 'cuda':
                pretrained_w = torch.load(pretrained_path)
            else:
                pretrained_w = torch.load(pretrained_path, map_location='cpu')
            
            print(f"Loading DeepVO pretrained model from {pretrained_path}")
            
            # Load the entire state dict
            model_dict = self.state_dict()
            
            # Filter pretrained weights to match DeepVO layers
            if 'state_dict' in pretrained_w:
                pretrained_dict = pretrained_w['state_dict']
            else:
                pretrained_dict = pretrained_w
                
            # Only update layers that exist in both models
            update_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    # Check if shapes match
                    if v.shape == model_dict[k].shape:
                        update_dict[k] = v
                        print(f"  Loading: {k} {v.shape}")
                    else:
                        print(f"  Skipping {k}: shape mismatch {v.shape} vs {model_dict[k].shape}")
            
            if update_dict:
                model_dict.update(update_dict)
                self.load_state_dict(model_dict)
                print(f"Successfully loaded {len(update_dict)} layers from pretrained DeepVO")
            else:
                print("Warning: No matching layers found in pretrained model")
                
        except Exception as e:
            print(f"Error loading pretrained DeepVO weights: {e}")
    
    def forward(self, x):
        # x: (batch, seq_len, channel, width, height)
        # stack consecutive image pairs
        x = torch.cat((x[:, :-1], x[:, 1:]), dim=2)
        batch_size = x.size(0)
        seq_len = x.size(1)
        # CNN
        x = x.view(batch_size*seq_len, x.size(2), x.size(3), x.size(4))
        x = self.encode_image(x)
        x = x.view(batch_size, seq_len, -1)
        
        # RNN
        out, hc = self.rnn(x)
        out = self.rnn_drop_out(out)
        # Two classification heads
        trans_logits = self.linear_trans(out)  # (batch, seq, num_classes)
        rot_logits = self.linear_rot(out)      # (batch, seq, num_classes)
        return trans_logits, rot_logits
    
    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6
    
    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]
    
    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]
    
    def get_loss(self, batch: Tuple[Any, ...]) -> torch.Tensor:
        """Compute loss for a batch following BaseModel interface."""
        _, x, y_trans_class, y_rot_class = batch
        # y_trans_class, y_rot_class should be (batch, seq) with class indices
        
        trans_logits, rot_logits = self.forward(x)
        # trans_logits, rot_logits: (batch, seq-1, num_classes)
        
        # we want labels for pairs (f0, f1), ..., (f_{T-2}, f_{T-1})
        y_trans_class = y_trans_class[:, 1:]  
        y_rot_class = y_rot_class[:, 1:]      
        
        # Reshape for cross_entropy: (batch*seq, num_classes) and (batch*seq,)
        batch_size, seq_len, num_classes = trans_logits.shape
        trans_logits_flat = trans_logits.reshape(-1, num_classes)
        rot_logits_flat = rot_logits.reshape(-1, num_classes)
        y_trans_flat = y_trans_class.reshape(-1).long()
        y_rot_flat = y_rot_class.reshape(-1).long()
        
        # Cross-entropy loss for classification
        translation_loss = torch.nn.functional.cross_entropy(trans_logits_flat, y_trans_flat)
        angle_loss = torch.nn.functional.cross_entropy(rot_logits_flat, y_rot_flat)
        
        # Weight angle loss higher (common in VO tasks)
        loss = (angle_loss + translation_loss)
        return loss
    
    def validation_step(self, batch: Tuple[Any, ...]) -> Dict[str, float]:
        """Validation step that computes loss and accuracy metrics."""
        _, x, y_trans_class, y_rot_class = batch
        
        with torch.no_grad():
            trans_logits, rot_logits = self.forward(x)
            
            # Align sequences
            y_trans_class = y_trans_class[:, 1:]
            y_rot_class = y_rot_class[:, 1:]
            
            # Flatten
            batch_size, seq_len, num_classes = trans_logits.shape
            trans_logits_flat = trans_logits.reshape(-1, num_classes)
            rot_logits_flat = rot_logits.reshape(-1, num_classes)
            y_trans_flat = y_trans_class.reshape(-1).long()
            y_rot_flat = y_rot_class.reshape(-1).long()
            
            # Compute loss
            translation_loss = torch.nn.functional.cross_entropy(trans_logits_flat, y_trans_flat)
            angle_loss = torch.nn.functional.cross_entropy(rot_logits_flat, y_rot_flat)
            loss = (100 * angle_loss + translation_loss)
            
            # Compute accuracies
            trans_pred = trans_logits_flat.argmax(dim=1)
            rot_pred = rot_logits_flat.argmax(dim=1)
            
            trans_acc = (trans_pred == y_trans_flat).float().mean()
            rot_acc = (rot_pred == y_rot_flat).float().mean()
            
            # Per-class accuracy
            trans_per_class_acc = []
            rot_per_class_acc = []
            for c in range(num_classes):
                mask_trans = y_trans_flat == c
                if mask_trans.sum() > 0:
                    trans_per_class_acc.append((trans_pred[mask_trans] == c).float().mean().item())
                
                mask_rot = y_rot_flat == c
                if mask_rot.sum() > 0:
                    rot_per_class_acc.append((rot_pred[mask_rot] == c).float().mean().item())
            
            metrics = {
                'val_loss': loss.item(),
                'val_trans_loss': translation_loss.item(),
                'val_rot_loss': angle_loss.item(),
                'val_trans_acc': trans_acc.item(),
                'val_rot_acc': rot_acc.item(),
            }
            
            # Add per-class accuracies
            for i, acc in enumerate(trans_per_class_acc):
                metrics[f'val_trans_acc_class_{i}'] = acc
            for i, acc in enumerate(rot_per_class_acc):
                metrics[f'val_rot_acc_class_{i}'] = acc
            
            return metrics