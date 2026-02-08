import glob
import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class DensityMapLuminanceDataset(Dataset):
    def __init__(self, image_luminance_dir, density_luminance_dir, norm_factor, transform=None):
        self.image_luminance_paths = sorted(glob.glob(os.path.join(image_luminance_dir, '*.npy')))
        self.density_luminance_paths = sorted(glob.glob(os.path.join(density_luminance_dir, '*.npy')))
        self.norm_factor = norm_factor
        self.transform = transform

    def __len__(self):
        return len(self.image_luminance_paths)

    def __getitem__(self, idx):
        image_luminance = np.load(self.image_luminance_paths[idx]).astype(np.float32)
        image_luminance = image_luminance / np.max(image_luminance)
        density_luminance = np.load(self.density_luminance_paths[idx]).astype(np.float32) / self.norm_factor
#        density_luminance = np.load(self.density_luminance_paths[idx]).astype(np.float32) / (1/1000)

        # Konvertieren in Tensoren
        image_luminance_tensor = torch.from_numpy(image_luminance).unsqueeze(0)
        density_luminance_tensor = torch.from_numpy(density_luminance).unsqueeze(0)

        if self.transform:
            image_luminance_tensor = self.transform(image_luminance_tensor)
            density_luminance_tensor = self.transform(density_luminance_tensor)

        return image_luminance_tensor, density_luminance_tensor

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True, padding_mode="replicate"), # kernelsize 3, stride 1, padding 1 (spiegelt Randwerte)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True, padding_mode="replicate"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        final_output = self.final_conv(x)
        return final_output

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.clone().detach().to(dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32)