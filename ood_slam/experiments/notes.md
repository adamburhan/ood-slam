
# Experiments (Jan 9 2026)

## Model
EarlyFusionResNet (num_images * 3 in_channels). For rgb image pair, 6 in_channels with the weights initialized as follows:

with torch.no_grad():
            n_repeats = in_channels // 3
            for i in range(n_repeats):
                new_conv1.weight[:, i*3:(i+1)*3] = original_weights / n_repeats