import numpy as np
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.optim as optim
from data_generation import (
    generate_normalized_amplitudes,
    generate_normalized_phase,
    load_mmf_modes_hdf5,
    generate_images,
    load_phase_variants_h5py,
    mmf_rebuilt_image_relative_phase,
    compute_correlation,
    plot_images,
)
from train import (
    MLP,
    train_model,
    plot_losses,
    test_model,
    CustomVGG_n5,
    criterion,
)
from torch.utils.data import DataLoader, Subset
from scipy.io import savemat
import gc
import mat73
import random

# Define parameters
number_of_modes = 5  # Number of modes
image_size = 32     # Resolution 32x32
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Load mode data
filename = 'mmf_5modes_32.mat'
mode_fields_mat = mat73.loadmat(filename)
mmf_modes = torch.tensor(mode_fields_mat["modes_field"])  # Complex array
mmf_modes = mmf_modes.to(device)  # Shape: (num_of_modes, image_size, image_size)

def read_predictions(filename):
    """
    Read prediction labels from a binary file and reshape them.

    Args:
        filename (str): Path to the binary file containing predictions.

    Returns:
        torch.Tensor: Tensor of predictions reshaped to (-1, 9).
    """
    predictions = np.fromfile(filename, dtype=np.float32)
    # Convert the NumPy array to a PyTorch tensor and reshape to the expected shape
    predictions_tensor = torch.tensor(predictions).reshape(-1, 9)
    return predictions_tensor

def read_images(filename):
    """
    Read image data from a binary file and reshape them.

    Args:
        filename (str): Path to the binary file containing images.

    Returns:
        torch.Tensor: Tensor of images reshaped to (-1, 1, 32, 32).
    """
    image_data = np.fromfile(filename, dtype=np.float32)
    # Reshape the NumPy array to a four-dimensional tensor [num_images, 1, 32, 32]
    image_tensor = torch.tensor(image_data).reshape(-1, 1, 32, 32)
    return image_tensor

if __name__ == '__main__':
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # Read predicted labels and reshape
    pred_filename = 'predicted_labels.bin'
    pred_vectors = read_predictions(pred_filename).to(device)

    # Read actual image data and reshape
    image_filename = 'test_images1.bin'
    ground_truth = read_images(image_filename).to(device)

    # Reconstruct images
    complex_vector_N, reconstructed_images, correlations, amplitude_vector, phase = mmf_rebuilt_image_relative_phase(
        pred_vectors, ground_truth, number_of_modes, mmf_modes, device, "rp", image_size
    )

    # Output the shape of the reconstructed images
    print("best_images.shape:", reconstructed_images.shape)  # Output the shape of reconstructed images

    # Calculate correlation coefficients
    correlations = compute_correlation(reconstructed_images, ground_truth)
    print("Correlation mean :", np.mean(correlations))

    # Calculate standard deviation of correlations
    correlation_std = torch.std(torch.tensor(correlations))
    print("Correlation Sd:", correlation_std.item())

    # Calculate the difference between reconstructed and true images
    difference = reconstructed_images - ground_truth
    # Calculate the standard deviation of the differences
    std_deviation = torch.std(difference)
    print("Correlation Standard Deviation:", std_deviation.item())

    # Select a random index for visualization
    best_index = random.randint(0, len(reconstructed_images) - 1)
    best_reconstructed_image = reconstructed_images[best_index].cpu().numpy().squeeze()
    best_ground_truth_image = ground_truth[best_index].cpu().numpy().squeeze()

    # Calculate standard deviation again (redundant, can be removed if not needed)
    difference = reconstructed_images - ground_truth
    std_deviation = torch.std(difference)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(best_ground_truth_image, cmap='viridis')
    axes[0].set_title("Original Image")
    axes[1].imshow(best_reconstructed_image, cmap='viridis')
    axes[1].set_title("Reconstructed Image")

    # Display correlation coefficient and standard deviation on the plot
    plt.suptitle(f'CC: {correlations[best_index]:.4f}, STD: {std_deviation.item():.4f}')
    plt.show()
