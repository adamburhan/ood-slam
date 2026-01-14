# Experiments (Jan 9 2026)
## Model
EarlyFusionResNet with num_images * 3 input channels. For RGB image pairs: 6 input channels.
Weight initialization (pretrained adaptation):
```python 
torch.no_grad():
    n_repeats = in_channels // 3
    for i in range(n_repeats):
        new_conv1.weight[:, i*3:(i+1)*3] = original_weights / n_repeats
```
Rationale: Pretrained ResNet expects 3 channels. To handle 6 channels while preserving pretrained feature detectors, we tile the original conv1 weights and divide by the number of repeats. This keeps the expected activation magnitude roughly constant at initialization (since input now sums over 2× as many channels).
## Loss
MSE on log-transformed targets: MSE(pred, log(target))
Rationale: 
## Results 

Val loss now decreasing (was flat before)—suggests previous overfitting was partly due to limited data
Train/val gap still exists (~0.2 vs ~1.25) but improved
Task metrics (mse_trans, mse_rot) show clear improvement
No significant difference between orb/rgb/combined modes

## Next steps
- TartanVO adapted to rpe prediction
- better ways to include orb features (either better representation or direct sufficient stats from instrumenting orbslam)