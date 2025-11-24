# from https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/master/model.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from typing import Tuple, Any
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
        
class DeepVO(BaseModel):
    def __init__(
        self, 
        img_h, 
        img_w, 
        clip, 
        conv_dropout,
        rnn_hidden_size,
        rnn_dropout_between,
        rnn_dropout_out,
        pretrained_flownet=None,
        batch_norm=True):
        super().__init__()
        # CNN
        self.batch_norm = batch_norm
        self.clip_grad_norm = clip  # Renamed for BaseModel compatibility
        self.pretrained_flownet = pretrained_flownet
        
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
            tmp = torch.zeros(1, 6, img_h, img_w, device=self.conv1.weight.device)
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
        self.linear = nn.Linear(in_features=rnn_hidden_size, out_features=6)
        
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
        if self.pretrained_flownet is not None:
            self._load_pretrained_flownet(self.pretrained_flownet)
    
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
    
    def forward(self, x):
        # x: (batch, seq_len, channel, width, height)
        # stack image
        x = torch.cat(x[:, :-1], x[:, 1:], dim=2)
        batch_size = x.size(0)
        seq_len = x.size(1)
        # CNN
        x = x.view(batch_size*seq_len, x.size(2), x.size(3), x.size(4))
        x = self.encode_image(x)
        x = x.view(batch_size, seq_len, -1)
        
        # RNN
        out, hc = self.rnn(x)
        out = self.rnn_drop_out(x)
        out = self.linear(out)
        return out
        
    
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
        _, x, y = batch
        predicted = self.forward(x)
        y = y[:, 1:, :] # (batch, seq, dim_pose)
        # Weighted MSE Loss
        angle_loss = torch.nn.functional.mse_loss(predicted[:, :, :3], y[:, :, :3])
        translation_loss = torch.nn.functional.mse_loss(predicted[:, :, 3:], y[:, :, 3:])
        loss = (100 * angle_loss + translation_loss)
        return loss