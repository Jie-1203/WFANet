import h5py
import time
import torch
import yaml
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from net_torch import HWViT
import scipy.io as sio
import math
import os
from thop import profile
from torchinfo import summary
import warnings
warnings.filterwarnings("ignore")


def learning_rate_function(x):
    return lr_max * math.exp((-1) * (math.log(2)) / 90 * x)

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

start_time = time.time()
device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
with open("super_para.yml") as f:
    super_para_dict = yaml.safe_load(f)
    lr_max, epochs, batch_size, ratio, weight_decay, pan_target_channel, ms_target_channel, head_channel, dropout = \
    super_para_dict["lr_max"], super_para_dict["epochs"], super_para_dict["batch_size"], super_para_dict['ratio'], \
    super_para_dict['weight_decay'], super_para_dict['pan_target_channel'], super_para_dict['ms_target_channel'], \
    super_para_dict['head_channel'], super_para_dict['dropout']


file_path = "Dataset/WV3/train_wv3-001.h5"
with h5py.File(file_path, 'r') as f:
    datasets = f.keys()
    for dataset_name in datasets:
        dataset = f[dataset_name]
        data = dataset[:]
        if dataset_name == "gt":
            gt = torch.from_numpy(np.array(data, dtype=np.float32))
        elif dataset_name == "pan":
            pan = torch.from_numpy(np.array(data, dtype=np.float32))
        elif dataset_name == "ms":
            ms = torch.from_numpy(np.array(data, dtype=np.float32)) 
        elif dataset_name == "lms":
            lms = torch.from_numpy(np.array(data, dtype=np.float32))
print("Sample data successfully fetched")
_, c, H, W = pan.shape
_, C, h, w = ms.shape

train_ds = TensorDataset(pan / ratio, gt / ratio, ms / ratio, lms / ratio)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

model = HWViT(L_up_channel=ms.shape[1], pan_channel=pan.shape[1], ms_target_channel=ms_target_channel,
              pan_target_channel=pan_target_channel, head_channel=head_channel, dropout=dropout)
model_optimizer = optim.Adam(
    [{'params': (p for name, p in model.named_parameters() if 'bias' not in name), 'weight_decay': weight_decay},
     {'params': (p for name, p in model.named_parameters() if 'bias' in name)}], lr=lr_max)
print(f'Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

# flops_pan = torch.rand(batch_size, c, H, W) 
# flops_ms = torch.rand(batch_size, C, h, w)
# flops_lms = torch.rand(batch_size, C, H, W)
# flops, _ = profile(model, inputs=(flops_pan, flops_ms, flops_lms), verbose=False)
# print(f"Model FLOPS: {int(flops)} FLOPS")

# summary(model, input_size=[(batch_size, c, H, W), (batch_size, C, h, w), (batch_size, C, H, W)])

criterion = nn.L1Loss(reduction='mean')
all_loss = []
print("Start training")
torch.cuda.empty_cache()
model.to(device)
for epoch in range(1, epochs + 1):
    single_loss = []
    for batch_idx, (pan, gt, ms, lms) in enumerate(train_loader):
        pan, gt, ms, lms = pan.to(device), gt.to(device), ms.to(device), lms.to(device)

        model.train()
        model_optimizer.zero_grad()

        output = model(pan=pan, ms=ms, lms=lms)
        loss = criterion(output, gt)
        loss.backward()

        model_optimizer.step()
        single_loss.append(loss)

    for param_group in model_optimizer.param_groups:
        param_group['lr'] = learning_rate_function(epoch)

    loss_number = sum(single_loss) / len(single_loss)
    all_loss.append(float(loss_number))
    print(f"Current epoch: {epoch}, loss: {loss_number}")
print("Training successful")
end_time = time.time()
print(f"Training time: {end_time - start_time}")

epoch = yaml.safe_load(open("super_para.yml"))["epochs"]
checkpoint_path = os.path.join("checkpoints", f'WFANet_epoch_{epoch}.pth')
torch.save({
    'model': model.state_dict(),
    'model_optimizer': model_optimizer.state_dict(),
    'epoch': epoch
}, checkpoint_path)
print("Model saved successfully")

################### Testing ###################

print('Start testing')
file_path = "Dataset/WV3/test_wv3_multiExm1.h5"
with h5py.File(file_path, 'r') as f:
    datasets = f.keys()
    for dataset_name in datasets:
        dataset = f[dataset_name]
        data = dataset[:]
        if dataset_name == "gt":
            test_gt = torch.from_numpy(np.array(data, dtype=np.float32))
        elif dataset_name == "pan":
            test_pan = torch.from_numpy(np.array(data, dtype=np.float32))
        elif dataset_name == "ms":
            test_ms = torch.from_numpy(np.array(data, dtype=np.float32))
        elif dataset_name == "lms":
            test_lms = torch.from_numpy(np.array(data, dtype=np.float32))

test_ds = TensorDataset(test_pan / ratio, test_gt / ratio, test_ms / ratio, test_lms / ratio)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
test_loss = []

for batch_idx, (test_pan, test_gt, test_ms, test_lms) in enumerate(test_loader):
    test_pan, test_gt, test_ms, test_lms = test_pan.to(device), test_gt.to(device), test_ms.to(device), test_lms.to(device)
    model.eval()
    with torch.no_grad():
        output = model(pan=test_pan, ms=test_ms, lms=test_lms)
        loss = criterion(output, test_gt)
        print(f"Test set {batch_idx} loss: {loss}")
        test_loss.append(float(loss))
        sio.savemat(os.path.join("results", f'output_mulExm_{batch_idx}.mat'), {f'sr': output.squeeze(0).cpu().numpy().transpose((1, 2, 0)) * ratio})

loss_data = {"train_loss": all_loss, "train_time": end_time - start_time, 'test_loss': test_loss}
with open(checkpoint_path, 'w') as file:
    yaml.dump(loss_data, file, default_flow_style=False)
