import pickle

file_path = "/home/adamb14/repos/ood-slam/logs/deepvo_error_reg_kitti/multiruns/2026-01-07/16-13-55/.submitit/55005602/55005602_0_result.pkl"
with open(file_path, 'rb') as f:
    data = pickle.load(f)
print(data)