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


# A function to split the input data for inference. If your GPU memory is insufficient, it is recommended to use this function for testing. 
def split_test(size, pad, main_folder, name, path):
    global device, ratio
    file_path = path
    with h5py.File(file_path, 'r') as f:
        datasets = f.keys()
        for dataset_name in datasets:
            dataset = f[dataset_name]
            data = dataset[:]
            if dataset_name == "pan":
                test_pan = torch.from_numpy(np.array(data, dtype=np.float32)) / ratio
            elif dataset_name == "ms":
                test_ms = torch.from_numpy(np.array(data, dtype=np.float32)) / ratio
            elif dataset_name == "lms":
                test_lms = torch.from_numpy(np.array(data, dtype=np.float32)) / ratio
    image_num, C, h, w = test_ms.shape
    _, _, H, W = test_pan.shape
    cut_size = size  # must be divided by 4, we recommend 64
    ms_size = cut_size // 4
    pad = pad  # must be divided by 4
    edge_H = cut_size - (H - (H // cut_size) * cut_size)
    edge_W = cut_size - (W - (W // cut_size) * cut_size)

    new_folder_1 = os.path.join(main_folder, name)
    os.makedirs(new_folder_1)
    for k in range(image_num):
        with torch.no_grad():
            x1, x2, x3 = test_ms[k, :, :, :], test_pan[k, 0, :, :], test_lms[k, :, :, :]
            x1 = x1.cpu().unsqueeze(dim=0).float()
            x2 = x2.cpu().unsqueeze(dim=0).unsqueeze(dim=1).float()
            x3 = x3.cpu().unsqueeze(dim=0).float()

            x1_pad = torch.zeros(1, C, h + pad // 2 + edge_H // 4, w + pad // 2 + edge_W // 4)
            x2_pad = torch.zeros(1, 1, H + pad * 2 + edge_H, W + pad * 2 + edge_W)
            x3_pad = torch.zeros(1, C, H + pad * 2 + edge_H, W + pad * 2 + edge_W)
            x1 = torch.nn.functional.pad(x1, (pad // 4, pad // 4, pad // 4, pad // 4), 'reflect')
            x2 = torch.nn.functional.pad(x2, (pad, pad, pad, pad), 'reflect')
            x3 = torch.nn.functional.pad(x3, (pad, pad, pad, pad), 'reflect')

            x1_pad[:, :, :h + pad // 2, :w + pad // 2] = x1
            x2_pad[:, :, :H + pad * 2, :W + pad * 2] = x2
            x3_pad[:, :, :H + pad * 2, :W + pad * 2] = x3
            output = torch.zeros(1, C, H + edge_H, W + edge_W)

            scale_H = (H + edge_H) // cut_size
            scale_W = (W + edge_W) // cut_size
            for i in range(scale_H):
                for j in range(scale_W):
                    MS = x1_pad[:, :, i * ms_size: (i + 1) * ms_size + pad // 2,
                         j * ms_size: (j + 1) * ms_size + pad // 2].to(device)
                    PAN = x2_pad[:, :, i * cut_size: (i + 1) * cut_size + 2 * pad,
                          j * cut_size: (j + 1) * cut_size + 2 * pad].to(device)
                    LMS = x3_pad[:, :, i * cut_size: (i + 1) * cut_size + 2 * pad,
                          j * cut_size: (j + 1) * cut_size + 2 * pad].to(device)
                    sr = model(pan=PAN, ms=MS, lms=LMS)
                    sr = torch.clamp(sr, 0, 1)
                    output[:, :, i * cut_size: (i + 1) * cut_size, j * cut_size: (j + 1) * cut_size] = \
                        sr[:, :, pad: cut_size + pad, pad: cut_size + pad] * ratio
            output = output[:, :, :H, :W]
            output = torch.squeeze(output).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
            new_path = os.path.join(new_folder_1, f'output_mulExm_{k}.mat')
            sio.savemat(new_path, {f'sr': output})
        print(k)

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

# A function to split the input data for inference. If your GPU memory is insufficient, it is recommended to use this function for testing. 
# split_test(64, 4, os.getcwd(), "DatasetWV3'", "../../Dataset/WV3/test_wv3_multiExm1.h5")
# split_test(128, 8, os.getcwd(), "DatasetWV3''", "../../Dataset/WV3/test_wv3_multiExm1.h5")

loss_data = {"train_loss": all_loss, "train_time": end_time - start_time, 'test_loss': test_loss}
with open(checkpoint_path, 'w') as file:
    yaml.dump(loss_data, file, default_flow_style=False)
