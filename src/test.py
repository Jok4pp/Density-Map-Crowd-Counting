# test.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
import sys

from model import *
from utils import *


# Definiere das Gerät (CPU oder GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(image_dir, model_path, output_dir, density_test_npy_dir, norm_factor, batch_size):
    print(f"Testing model {model_path}")

    transform = transforms.Compose([
        transforms.Lambda(to_tensor)
    ])

    dataset = DensityMapLuminanceDataset(image_dir, image_dir, norm_factor, transform)
    model = UNET()

    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8,  
        pin_memory=True  
    )

    model_ref = torch.load(model_path, weights_only=True)
    model.load_state_dict(model_ref['model_state_dict'])
    model = model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    tiff_dir = os.path.join(output_dir, 'Tiff')
    os.makedirs(tiff_dir, exist_ok=True)
    numpy_dir = os.path.join(output_dir, '.npy')
    os.makedirs(numpy_dir, exist_ok=True)

    predicted_density_sums = []
    gt_density_sums = [] 
    print("Calculating GT density sums...")
    for file_name in sorted(os.listdir(density_test_npy_dir)):
        if file_name.endswith('.npy'):
            file_path = os.path.join(density_test_npy_dir, file_name)
            gt_density = np.load(file_path)
            gt_sum = np.sum(gt_density)
            gt_density_sums.append(gt_sum)
            print(f"Processed GT file {file_name}: Sum = {gt_sum:.6f}")

    print("Running model predictions...")
    image_count = 1

    with torch.no_grad():
        for idx, (image, density) in enumerate(data_loader):
            image = image.to(device)
            output = model(image)

            
            output = output.squeeze().cpu().numpy()
            density = density.squeeze().cpu().numpy()
            output = np.clip(output, 0, 1)

            if output.ndim == 2:
                image_number = 1
            else: 
                image_number = output.shape[0]

            for i in range(image_number):
                # Greife auf jedes Bild im Batch zu

                if output.ndim == 3:  # Nach 4-Dimensionalität prüfen
                    single_output = output[i].squeeze() * norm_factor
                else: 
                    single_output = output * norm_factor

                # Numpy speichern
                output_path_npy = os.path.join(numpy_dir, f"pred_lum_{image_count:04d}.npy")
                np.save(output_path_npy, single_output)


                # Crowdgröße speichern
                density_sum = np.sum(single_output)
                predicted_density_sums.append(density_sum)
                print(f"Processed Test file {image_count}: Sum = {density_sum:.6f}")


                # Tiff speichern
                max_value = np.max(single_output)
                if max_value > 0 and not np.isnan(max_value):
                    density_map_8bit = (255 * (single_output / max_value)).astype(np.uint8)
                else:
                    print(f"Warning: Output {image_count} has invalid or zero max value. Skipping normalization.")
                    density_map_8bit = np.zeros_like(single_output, dtype=np.uint8)  # Default to black image


                output_path_tiff = os.path.join(tiff_dir, f"pred_lum_{image_count:04d}.tiff")
                density_map_8bit = (255 * (single_output / np.max(single_output))).astype(np.uint8)
                img = Image.fromarray(density_map_8bit)
                img.save(output_path_tiff, format='TIFF')

                image_count += 1

    # Speicherung der Dichtesummen
    density_sum_path = os.path.join(output_dir, "predicted_density_sums.txt")
    np.savetxt(density_sum_path, predicted_density_sums, fmt='%.6f')
    print(f"Saved predicted density sums to {density_sum_path}")

    gt_sum_path = os.path.join(output_dir, "gt_density_sums.txt")
    np.savetxt(gt_sum_path, gt_density_sums, fmt='%.1f')
    print(f"Saved ground-truth density sums to {gt_sum_path}")

    # Berechnung der prozentualen Abweichung
    gt_count = np.loadtxt(gt_sum_path)
    pred_count = np.loadtxt(density_sum_path)
    
    # Exclude images with gt_count = 0
    valid_indices = gt_count != 0
    valid_gt_count = gt_count[valid_indices]
    valid_pred_count = pred_count[valid_indices]

    percent_deviation = np.abs((valid_pred_count - valid_gt_count) / valid_gt_count) * 100

    std_deviation = np.std(percent_deviation)
    mean_deviation = np.mean(percent_deviation)
    percentage_path = os.path.join(output_dir, "percentage.txt")
    i = 0
    with open(percentage_path, 'w') as f:
        for idx, deviation in enumerate(percent_deviation):
            i = idx + 1
            f.write(f"Abweichung für Testbild {i}: {deviation:.2f} %\n")
        f.write(f"\nMittelwert der Abweichungen: {mean_deviation:.2f} %\n")
        f.write(f"Standardabweichung: {std_deviation:.2f} %\n")
        f.write(f"Anzahl der Bilder mit gt_count = 0: {len(gt_count) - len(valid_gt_count)}\n")

    print("Finished Testing")