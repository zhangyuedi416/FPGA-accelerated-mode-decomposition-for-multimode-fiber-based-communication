import torch
import numpy as np
# from torchvision.transforms import Normalize
import h5py
# from torchvision.utils import save_image
# import os
from corr_coeff import corr2_torch
import matplotlib.pyplot as plt

# Set the device to CUDA if available, otherwise CPU
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Generate random amplitude weights and normalize them
def generate_normalized_amplitudes(num_modes, num_data):
    amplitude_weights = torch.randn(num_data, num_modes)    # Generate random numbers with standard normal distribution (mean=0, std=1)
    amplitude_squared = torch.pow(amplitude_weights, 2)     # Calculate the square of the absolute value of each amplitude
    # Normalize the squared amplitudes for each sample so that the sum of squares equals 1
    normalized_amplitudes_squared = amplitude_squared / torch.sum(amplitude_squared, dim=1, keepdim=True)
    # Take the square root to normalize the amplitudes
    normalized_amplitudes = torch.sqrt(normalized_amplitudes_squared)
    return normalized_amplitudes

# Generate normalized phases
def generate_normalized_phase(num_modes, num_data):
    # Initialize a tensor to store phases, all initialized to 0
    phases = torch.zeros(num_data, num_modes)

    # Generate random phases between 0 and 1 for the second mode
    phases[:, 1] = torch.rand(num_data)

    # Generate random phases between 0 and 2 for the third and subsequent modes
    for i in range(2, num_modes):
        phases[:, i] = torch.rand(num_data) * 2

    # Create label_phases by dividing each element of phases by 2 (except the first column)
    label_phases = phases.clone()
    label_phases[:, 1:] /= 2

    return phases, label_phases

# Generate normalized relative differences
def generate_normalized_relative_difference(num_modes, num_data):
    # Generate relative phase differences in the range [-π, π), starting from the second mode
    relative_phases = (torch.rand(num_data, num_modes - 1) * 2 - 1)
    # Set the phase of the first mode to zero
    first_mode_phase = torch.zeros(num_data, 1)
    # Concatenate the relative phase differences with the first mode's phase
    phases = torch.cat((first_mode_phase, relative_phases), dim=1)
    return phases

# Generate the dataset
def generate_dataset(number_of_data, number_of_modes):
    # Generate complex mode weights and label vectors
    amplitudes = generate_normalized_amplitudes(number_of_modes, number_of_data)
    scaled_cos_phases, phases = generate_normalized_phase_with_relative_difference(number_of_modes, number_of_data)

    # Construct label vectors (amplitudes + normalized cosine phases)
    labels = torch.cat((amplitudes, scaled_cos_phases[:, 1:]), dim=1)
    return amplitudes, scaled_cos_phases, phases, labels

# Load MMF modes from an HDF5 file
def load_mmf_modes_hdf5(filename, number_of_modes):
    with h5py.File(filename, 'r') as f:
        print(list(f.keys()))  # Print the top-level group names in the file
        if number_of_modes == 3:
            # Read real and imaginary parts
            real_part = f['mmf_3modes_32']['real'][()]
            imag_part = f['mmf_3modes_32']['imag'][()]
        elif number_of_modes == 5:
            real_part = f['mmf_5modes_32']['real'][()]
            imag_part = f['mmf_5modes_32']['imag'][()]
        elif number_of_modes == 10:
            real_part = f['modes_field']['real'][()]
            imag_part = f['modes_field']['imag'][()]
            real_part = np.transpose(real_part, (2, 0, 1))
            imag_part = np.transpose(imag_part, (2, 0, 1))
    # Combine real and imaginary parts to create a complex array
    mmf_modes = real_part + 1j * imag_part
    # mmf_modes = mmf_modes * brightness_factor
    min_val = np.min(mmf_modes)
    max_val = np.max(mmf_modes)
    normalized_mmf_modes = (mmf_modes) / (max_val)

    return torch.tensor(normalized_mmf_modes, dtype=torch.complex64)

# Load phase variants from an HDF5 file
def load_phase_variants_h5py(file_name, number_of_modes):
    with h5py.File(file_name, 'r') as file:
        if 'phase_weight' in file:
            # Read the data
            phase_weight_data = file['phase_weight'][()]

            # Convert the data to a Numpy array
            phase_weight_data = np.array(phase_weight_data)

            # Transpose the data to match Matlab's column-major layout before converting to PyTorch tensor
            phase_weight_data = phase_weight_data.T

            # Convert the Numpy array to a PyTorch tensor
            phase_weight_tensor = torch.tensor(phase_weight_data, dtype=torch.complex64)

            print(phase_weight_tensor)
            return phase_weight_tensor
        else:
            print("Dataset 'phase_weight' not found in the file.")
            return None

# Function to generate image data
def generate_images(mmf_modes, complex_weights_vector, image_size, number_of_data, model_type):
    # Create a four-dimensional tensor to store generated images
    image_data = torch.zeros((number_of_data, 1, image_size, image_size), dtype=torch.float32).to(device)
    # Initialize amplitude_distribution with the correct shape and type
    amplitude_distribution = torch.zeros((number_of_data, image_size, image_size), dtype=torch.float32).to(device)
    for index in range(number_of_data):
        complex_field = torch.zeros((image_size, image_size), dtype=torch.complex64).to(device)  # Initialize complex field
        # Construct the complex field by multiplying each mode with its corresponding weight. The contribution of each mode depends on its pattern and corresponding weight.
        for mode in range(mmf_modes.shape[0]):
            mode_pattern = mmf_modes[mode, :, :]    # Get the current mode pattern
            weight = complex_weights_vector[index, mode]    # Get the current mode's weight
            complex_field += mode_pattern * weight  # Accumulate into the complex field.
            # complex_field contains the cumulative contributions of all modes, each proportional to its corresponding complex weight.

        amplitude_distribution[index, :, :] = torch.abs(complex_field)  # Save the amplitude distribution
        # Get the current amplitude distribution and add a channel dimension to make it [1, H, W]
        current_amplitude_distribution = amplitude_distribution[index, :, :].unsqueeze(0)

        # Min-max normalize the amplitude distribution to the [0, 1] range
        min_val = current_amplitude_distribution.min()
        max_val = current_amplitude_distribution.max()
        normalized_image = (current_amplitude_distribution - min_val) / (max_val - min_val)
        image_data[index] = normalized_image
        if index < 50 and model_type != "original":
            image_to_save = image_data[index]
            # brightness_factor = 100  # Adjust image brightness
            # # image_to_save = image_to_save * brightness_factor
            image_to_save = torch.clamp(image_to_save, min=0, max=1)  # Ensure values are within [0, 1]

            image_to_save = image_to_save.cpu()     # Move image data to CPU
            min_value = image_to_save.min()
            max_value = image_to_save.max()
            with open('min_max_values.txt', 'w') as file:   # Write min and max values to a file
                file.write(f"Minimum Value: {min_value}\n")
                file.write(f"Maximum Value: {max_value}\n")
            save_directory = 'generated_image'
            # save_image(image_to_save, os.path.join(save_directory, f'generated_image_{index}_{model_type}.png'))
    return image_data, amplitude_distribution

# Rebuild image from MMF using predicted vectors and ground truth
def mmf_rebuilt_image(pred_vectors, ground_truth, number_of_modes, phase_variants, mmf_modes, device, model_type, image_size):
    number_of_test_images = pred_vectors.shape[0]
    ground_truth = ground_truth.squeeze()
    complex_vector_N = torch.zeros((number_of_test_images, number_of_modes), dtype=torch.complex64, device=device)
    phase_js = torch.zeros((number_of_test_images, number_of_modes), dtype=torch.complex64, device=device)
    max_correlations = []
    for i1 in range(number_of_test_images):
        # Read amplitude weights
        amplitude_vector = pred_vectors[i1, :number_of_modes]
        # Read cosine phases and normalize to (-1, 1)
        cos_phase = pred_vectors[i1, number_of_modes:]
        cos_phase_normalized = cos_phase * 2 - 1  # Normalize cos(phase) to (-1, 1)
        phase = torch.arccos(cos_phase_normalized)  # Compute phase using arccos()
        phase = torch.cat((torch.zeros(1, device=device), phase))  # Add phase weight for the first mode (phase value = 0)
        # Generate all possible phase combinations
        phi_vectors = phase.unsqueeze(0) * phase_variants
        complex_vector_n = torch.zeros_like(phi_vectors, dtype=torch.complex64, device=device)
        ground_truth_i = ground_truth[i1, :, :]
        correlation_n = torch.zeros(phi_vectors.size(0), device=device)

        # Reconstruct all possible field distributions
        for i2 in range(phi_vectors.size(0)):
            complex_vector = amplitude_vector * torch.exp(1j * phi_vectors[i2, :])
            template = torch.zeros_like(ground_truth_i, dtype=torch.complex64, device=device)

            for mode in range(number_of_modes):
                template += mmf_modes[mode, :, :] * complex_vector[mode]

            correlation = torch.abs(corr2_torch(template.abs(), ground_truth_i))
            correlation = corr2_torch(template.abs(), ground_truth_i)
            # correlation = torch.abs(corr2_torch(template, ground_truth_i))
            correlation_n[i2] = correlation
            complex_vector_n[i2, :] = complex_vector

        # Find the correct phase weights with the maximum correlation
        posx = torch.argmax(correlation_n)  # Find the position of maximum correlation
        complex_vector_N[i1, :] = complex_vector_n[posx, :].to(device)
        phase_js[i1, :] = phi_vectors[posx, :].to(device)
        # Save the maximum correlation value for the current image
        max_correlations.append(correlation_n[posx].item())

    reconstructed_images, rebuild_amplitude_distribution = generate_images(mmf_modes, complex_vector_N, image_size, complex_vector_N.shape[0], model_type)
    return complex_vector_N, reconstructed_images, max_correlations, amplitude_vector, phase_js

# Compute correlation between predicted and true images
def compute_correlation(pred_images, true_images):
    correlations = []
    for pred, true in zip(pred_images, true_images):
        correlation = corr2_torch(pred, true)
        correlations.append(correlation.item())
    return correlations

# Plot images with titles and correlations
def plot_images(images_list, selected_indices, titles, overall_title, correlations_list):
    num_rows = len(selected_indices)
    num_cols = len(images_list)

    plt.figure(figsize=(num_cols * 5, num_rows * 5))
    plt.suptitle(overall_title)

    for row_index, img_index in enumerate(selected_indices):
        for col_index, images in enumerate(images_list):
            ax = plt.subplot(num_rows, num_cols, row_index * num_cols + col_index + 1)
            plt.subplot(num_rows, num_cols, row_index * num_cols + col_index + 1)
            image = images[img_index].squeeze()
            if image.is_cuda:
                image = image.cpu()
            plt.imshow(image, cmap='gray')
            plt.title(titles[col_index])  # Add title to each image
            plt.axis('off')

            # If not the first group of images, display the correlation value
            if col_index != 0:  # Skip the first group of images
                correlation = correlations_list[col_index - 1][img_index]
                plt.text(0.5, -0.15, f'Correlation: {correlation:.4f}', ha='center', va='center', transform=ax.transAxes)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig(f'{overall_title}.png')
    plt.close()  # Close the figure to prevent resource usage in the background
    # plt.show()

# Rebuild image without using relative differences
def mmf_rebuilt_image_n(pred_vectors, ground_truth, number_of_modes, mmf_modes, device, model_type, image_size):
    number_of_test_images = pred_vectors.shape[0]
    ground_truth = ground_truth.squeeze()
    complex_vector_N = torch.zeros((number_of_test_images, number_of_modes), dtype=torch.complex64, device=device)
    max_correlations = []
    for i1 in range(number_of_test_images):
        # Read amplitude weights
        amplitude_vector = pred_vectors[i1, :number_of_modes]
        # Read cosine phases and normalize to (-1, 1)
        phase = pred_vectors[i1, number_of_modes:] * np.pi
        phase = torch.cat((torch.zeros(1, device=device), phase))  # Add phase weight for the first mode (phase value = 0)
        complex_vector_n = torch.zeros_like(phase, dtype=torch.complex64, device=device)
        ground_truth_i = ground_truth[i1, :, :]
        # correlation_n = torch.zeros(phase.size(0), device=device)

        complex_vector_N[i1, :] = amplitude_vector * torch.exp(1j * phase).to(device)
        template = torch.zeros_like(ground_truth_i, dtype=torch.complex64, device=device)

        for mode in range(number_of_modes):
            template += mmf_modes[mode, :, :] * complex_vector_N[i1, mode]

        correlation_n = corr2_torch(template.abs(), ground_truth_i)

        # Save the maximum correlation value for the current image
        max_correlations.append(correlation_n.item())

    reconstructed_images, rebuild_amplitude_distribution = generate_images(mmf_modes, complex_vector_N, image_size, complex_vector_N.shape[0], model_type)
    return complex_vector_N, reconstructed_images, max_correlations

# Rebuild image using absolute phase
def mmf_rebuilt_image_absolute_phase(pred_vectors, ground_truth, number_of_modes, phase_variants, mmf_modes, device, model_type, image_size):
    number_of_test_images = pred_vectors.shape[0]
    ground_truth = ground_truth.squeeze()
    complex_vector_N = torch.zeros((number_of_test_images, number_of_modes), dtype=torch.complex64, device=device)
    max_correlations = []
    for i1 in range(number_of_test_images):
        # Read amplitude weights
        amplitude_vector = pred_vectors[i1, :number_of_modes]
        phase = pred_vectors[i1, number_of_modes:]

        # Generate all possible phase combinations
        phi_vectors = phase * phase_variants
        complex_vector_n = torch.zeros_like(phi_vectors, dtype=torch.complex64, device=device)
        ground_truth_i = ground_truth[i1, :, :]
        correlation_n = torch.zeros(phi_vectors.size(0), device=device)

        # Reconstruct all possible field distributions
        for i2 in range(phi_vectors.size(0)):
            complex_vector = amplitude_vector * torch.exp(1j * phi_vectors[i2, :])
            template = torch.zeros_like(ground_truth_i, dtype=torch.complex64, device=device)

            for mode in range(number_of_modes):
                template += mmf_modes[mode, :, :] * complex_vector[mode]

            correlation = torch.abs(corr2_torch(template.abs(), ground_truth_i))
            correlation = corr2_torch(template.abs(), ground_truth_i)
            # correlation = torch.abs(corr2_torch(template, ground_truth_i))
            correlation_n[i2] = correlation
            complex_vector_n[i2, :] = complex_vector

        # Find the correct phase weights with the maximum correlation
        posx = torch.argmax(correlation_n)  # Find the position of maximum correlation
        complex_vector_N[i1, :] = complex_vector_n[posx, :].to(device)
        # Save the maximum correlation value for the current image
        max_correlations.append(correlation_n[posx].item())

    reconstructed_images, rebuild_amplitude_distribution = generate_images(mmf_modes, complex_vector_N, image_size, complex_vector_N.shape[0], model_type)
    return complex_vector_N, reconstructed_images, max_correlations

# Rebuild image using relative phase
def mmf_rebuilt_image_relative_phase(pred_vectors, ground_truth, number_of_modes, mmf_modes, device, model_type, image_size):
    number_of_test_images = pred_vectors.shape[0]
    ground_truth = ground_truth.squeeze()
    # labelphase except 2, now needs to be restored
    pred_vectors[:, number_of_modes:] *= 2

    complex_vector_N = torch.zeros((number_of_test_images, number_of_modes), dtype=torch.complex64, device=device)
    phase_N = torch.zeros((number_of_test_images, number_of_modes), dtype=torch.complex64, device=device)
    max_correlations = []
    for i1 in range(number_of_test_images):
        # Read amplitude weights
        amplitude_vector = pred_vectors[i1, :number_of_modes]
        phase = pred_vectors[i1, number_of_modes:] * np.pi
        phase = torch.cat((torch.zeros(1, device=device), phase))  # Add phase weight for the first mode (phase value = 0)
        phase_N[i1] = phase
        amplitude_N = pred_vectors[:, :number_of_modes].to(device)
        # Generate all possible phase combinations
        phi_vectors = phase.unsqueeze(0)
        complex_vector_n = torch.zeros_like(phi_vectors, dtype=torch.complex64, device=device)
        ground_truth_i = ground_truth[i1, :, :]
        correlation_n = torch.zeros(phi_vectors.size(0), device=device)

        # Reconstruct all possible field distributions
        for i2 in range(phi_vectors.size(0)):
            complex_vector = amplitude_vector * torch.exp(1j * phi_vectors[i2, :])
            template = torch.zeros_like(ground_truth_i, dtype=torch.complex64, device=device)

            for mode in range(number_of_modes):
                template += mmf_modes[mode, :, :] * complex_vector[mode]

            correlation = torch.abs(corr2_torch(template.abs(), ground_truth_i))
            correlation = corr2_torch(template.abs(), ground_truth_i)
            # correlation = torch.abs(corr2_torch(template, ground_truth_i))
            correlation_n[i2] = correlation
            complex_vector_n[i2, :] = complex_vector

        # Find the correct phase weights with the maximum correlation
        posx = torch.argmax(correlation_n)  # Find the position of maximum correlation
        complex_vector_N[i1, :] = complex_vector_n[posx, :].to(device)
        # Save the maximum correlation value for the current image
        max_correlations.append(correlation_n[posx].item())

    reconstructed_images, rebuild_amplitude_distribution = generate_images(mmf_modes, complex_vector_N, image_size, complex_vector_N.shape[0], model_type)

    return complex_vector_N, reconstructed_images, max_correlations, amplitude_N, phase_N

# Rebuild image (final version)
def mmf_rebuilt_image(pred_vectors, ground_truth, number_of_modes, mmf_modes, device, model_type, image_size):
    number_of_test_images = pred_vectors.shape[0]
    ground_truth = ground_truth.squeeze()
    pred_vectors[:, number_of_modes:] *= 2

    complex_vector_N = torch.zeros((number_of_test_images, number_of_modes), dtype=torch.complex64, device=device)
    phase_N = torch.zeros((number_of_test_images, number_of_modes), dtype=torch.complex64, device=device)
    max_correlations = []
    for i1 in range(number_of_test_images):
        amplitude_vector = pred_vectors[i1, :number_of_modes]
        phase = pred_vectors[i1, number_of_modes:] * np.pi
        phase = torch.cat((torch.zeros(1, device=device), phase))
        phase_N[i1] = phase

        complex_vector_N[i1, :] = amplitude_vector * torch.exp(1j * phase).to(device)

    reconstructed_images, rebuild_amplitude_distribution = generate_images(mmf_modes, complex_vector_N, image_size, complex_vector_N.shape[0], model_type)

    return reconstructed_images, range(number_of_test_images)

# Rebuild image without using relative differences
def mmf_rebuilt_image_without_relative_difference(pred_vectors, ground_truth, number_of_modes, phase_variants, mmf_modes, device, model_type, image_size):
    number_of_test_images = pred_vectors.shape[0]
    ground_truth = ground_truth.squeeze()
    complex_vector_N = torch.zeros((number_of_test_images, number_of_modes), dtype=torch.complex64, device=device)
    max_correlations = []
    for i1 in range(number_of_test_images):
        # Read amplitude weights
        amplitude_vector = pred_vectors[i1, :number_of_modes]
        # Read cosine phases and normalize to (-1, 1)
        cos_phase = pred_vectors[i1, number_of_modes:]
        cos_phase_normalized = cos_phase * 2 - 1  # Normalize cos(phase) to (-1, 1)
        phase = torch.arccos(cos_phase_normalized)  # Compute phase using arccos()

        # Generate all possible phase combinations
        phi_vectors = phase * phase_variants
        complex_vector_n = torch.zeros_like(phi_vectors, dtype=torch.complex64, device=device)
        ground_truth_i = ground_truth[i1, :, :]
        correlation_n = torch.zeros(phi_vectors.size(0), device=device)

        # Reconstruct all possible field distributions
        for i2 in range(phi_vectors.size(0)):
            complex_vector = amplitude_vector * torch.exp(1j * phi_vectors[i2, :])
            template = torch.zeros_like(ground_truth_i, dtype=torch.complex64, device=device)

            for mode in range(number_of_modes):
                template += mmf_modes[mode, :, :] * complex_vector[mode]

            correlation = torch.abs(corr2_torch(template.abs(), ground_truth_i))
            # correlation = torch.abs(corr2_torch(template, ground_truth_i))
            correlation_n[i2] = correlation
            complex_vector_n[i2, :] = complex_vector

        # Find the correct phase weights with the maximum correlation
        posx = torch.argmax(correlation_n)  # Find the position of maximum correlation
        complex_vector_N[i1, :] = complex_vector_n[posx, :].to(device)
        # Save the maximum correlation value for the current image
        max_correlations.append(correlation_n[posx].item())

    reconstructed_images, rebuild_amplitude_distribution = generate_images(mmf_modes, complex_vector_N, image_size, complex_vector_N.shape[0], model_type)
    return complex_vector_N, reconstructed_images, max_correlations
