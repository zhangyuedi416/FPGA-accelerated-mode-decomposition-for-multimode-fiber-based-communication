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
    CustomVGG_n,
    CustomVGG_n5,
    criterion,
)
from torch.utils.data import DataLoader, Subset
from torch.quantization import QuantStub, DeQuantStub, prepare, convert
from torch.quantization.qconfig import QConfig
from torch.quantization.observer import MinMaxObserver
from scipy.io import savemat
import scipy.io as sio
import gc
import mat73
import random

# Define parameters
number_of_modes = 3  # Number of modes
number_of_data = 42000
image_size = 16  # Resolution 16x16
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
num_epochs = 1000

# Generate complex mode weights and label vectors
amplitudes = generate_normalized_amplitudes(number_of_modes, number_of_data)
phases, label_phases = generate_normalized_phase(number_of_modes, number_of_data)

# Construct label vectors (amplitudes + normalized cosine phases)
# Exclude the first phase element to make it 2N-1, the second and third dimensions are based on relative differences calculated from the first dimension
labels = torch.cat((amplitudes, label_phases[:, 1:]), dim=1)

# Split data into training, validation, and test sets
val_size = 1000
test_size = 1000
train_size = number_of_data - test_size - val_size

# Generate random indices for dataset partitioning
indices = torch.randperm(number_of_data)
train_indices = indices[:train_size]
val_indices = indices[train_size : train_size + val_size]
test_indices = indices[train_size + val_size :]

# Partition the dataset based on indices
train_labels = labels[train_indices]
val_labels = labels[val_indices]
test_labels = labels[test_indices]

print(f"Train Labels Shape: {train_labels.shape}")
print(f"Validation Labels Shape: {val_labels.shape}")
print(f"Test Labels Shape: {test_labels.shape}")

# Load mode data
filename = 'mmf_3modes_16.mat'
mode_fields_mat = mat73.loadmat(filename)
mmf_modes = torch.tensor(mode_fields_mat["modes_field"])  # Complex array
mmf_modes = mmf_modes.to(device)  # (num_of_modes, image_size, image_size)

# Generate image data
# Convert phases from normalized cosine values back to original phase values
original_phases = phases * np.pi

# For each data point, calculate complex weights ki = ρi * e^(iϕi) for each mode, where e^iθ = cos(θ) + i * sin(θ)
complex_weights_vector = amplitudes * torch.exp(1j * original_phases)
complex_weights_vector = complex_weights_vector.to(device)

# Generate images
image_data, amplitude_distribution = generate_images(
    mmf_modes, complex_weights_vector, image_size, number_of_data, "original"
)
# Get corresponding data for the test set
amplitude_distribution_test = amplitude_distribution[test_indices]
image_data_test = image_data[test_indices]
complex_weights_vector_test = complex_weights_vector[test_indices]
number_of_data_test = test_size

# Generate test set images
image_data_test_2, amplitude_distribution_test_2 = generate_images(
    mmf_modes, complex_weights_vector_test, image_size, number_of_data_test, "test"
)

# Create datasets based on image_data and labels
train_dataset = TensorDataset(image_data[train_indices], train_labels)
val_dataset = TensorDataset(image_data[val_indices], val_labels)
test_dataset = TensorDataset(image_data[test_indices], test_labels)

# Get all test images and labels
test_images_tensor, test_labels_tensor = test_dataset.tensors

# Ensure tensors are on CPU
test_images_tensor = test_images_tensor.cpu()
test_labels_tensor = test_labels_tensor.cpu()

# Convert to NumPy arrays
test_images_np = test_images_tensor.numpy()  # Shape is [1000, 1, 16, 16]
test_labels_np = test_labels_tensor.numpy()  # Shape is [1000, 5], assuming each sample has 5 labels

# Save images and labels as binary files
test_images_np.tofile('test_images.bin')
test_labels_np.tofile('test_labels.bin')

# Define batch size and learning rate
batch_size = 258
learning_rate = 0.0006

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize VGG
input_size = 16 * 16  # Input size 16*16 = 256
output_size = train_labels.shape[1]
# model = MLP(input_size, output_size).to(device)
model_VGG_3_modes = CustomVGG_n(output_features=2 * number_of_modes - 1).to(device)
model = model_VGG_3_modes

# Define loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Add weight_decay parameter
# Define learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

# Start training
train_losses, val_losses = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs
)
plot_losses(train_losses, val_losses, 'relative positive')

def save_losses(train_losses, val_losses, filename='losses.mat'):
    losses_dict = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    sio.savemat(filename, losses_dict)

# Call the function to save data
save_losses(train_losses, val_losses)

# Save the trained model weights
torch.save(model.state_dict(), 'trained_model.pth')

# Test VGG and get predictions and actual labels
pred_vectors, true_labels = test_model(model, test_loader)

# Define predictions and ground truth
ground_truth = image_data_test
pred_vectors = pred_vectors.to(device)
ground_truth = ground_truth.to(device)

# Calculate mean absolute error
absolute_errors = torch.abs(pred_vectors - true_labels.to(device))
mean_absolute_error = torch.mean(absolute_errors)
print(f"Mean Absolute Error: {mean_absolute_error}")

# Calculate mean absolute error for amplitudes and phases separately
amplitude_errors = absolute_errors[:, :number_of_modes]
phase_errors = absolute_errors[:, number_of_modes:]
mean_amplitude_error = torch.mean(amplitude_errors)
mean_phase_error = torch.mean(phase_errors)
print(f"Mean Amplitude Error: {mean_amplitude_error}")
print(f"Mean Phase Error: {mean_phase_error}")

# Reconstruct images
complex_vector_N, reconstructed_images, correlations, amplitude_vector, phase = mmf_rebuilt_image_relative_phase(
    pred_vectors, ground_truth, number_of_modes, mmf_modes, device, "rp", image_size
)

# Output the shape of the best images
print("best_images.shape:", reconstructed_images.shape)

# Find the indices of the top 10 images with the highest correlations
best_indices = np.argsort(correlations)[-10:]

# Extract the top 10 original and corresponding reconstructed images
best_ground_truth_images = ground_truth[best_indices].cpu().numpy()
best_reconstructed_images = reconstructed_images[best_indices].cpu().numpy()
best_ground_truth_images = best_ground_truth_images.squeeze()
best_reconstructed_images = best_reconstructed_images.squeeze()

image_data_dict = {
    'best_ground_truth_images': best_ground_truth_images,  # Original images
    'best_reconstructed_images': best_reconstructed_images,  # Reconstructed images
}

# Save as a .mat file using savemat
savemat('best_images_data.mat', image_data_dict)

print("ok")

# Calculate correlation coefficients
pred_images = reconstructed_images

all_true_images = []

for images, _ in test_loader:
    all_true_images.append(images)

# Concatenate all batches into a single tensor
true_images = torch.cat(all_true_images, dim=0)

# Compute correlation coefficients
correlations = compute_correlation(reconstructed_images, ground_truth)
print("Correlation mean :", np.mean(correlations))

# Calculate standard deviation
correlation_std = torch.std(torch.tensor(correlations))
print("Correlation Standard Deviation:", correlation_std.item())

# Save model parameters to binary files
for name, param in model.named_parameters():
    # First, ensure the parameter is on CPU
    param_cpu = param.detach().cpu().numpy()  # Use .detach() to get the parameter, .cpu() to ensure it's on CPU, then convert to numpy
    param_cpu.tofile("{}.bin".format(name.replace('.', '_')))  # Save the parameter to a binary file

# Clean up
del model
gc.collect()  # Call the garbage collector
if torch.cuda.is_available():
    torch.cuda.empty_cache()
