import numpy as np
from torch import nn
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Define a hook function to store each layer's output
layer_outputs = {}  # Dictionary to store outputs of each layer

def get_layer_output(module, input, output):
    layer_name = str(module)
    layer_outputs[layer_name] = output.detach()

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),  # Added layers to make the model deeper
            nn.ReLU(),
            nn.Dropout(0.5),  # Added Dropout layer to reduce overfitting
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input data
        return self.layers(x)

criterion = nn.MSELoss()  # Mean Squared Error Loss function

# Train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30):
    early_stopping = EarlyStopping(patience=10, verbose=True, path='early_stopping_model.pth')
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero gradients
            outputs = model(inputs)  # Compute model output
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            train_loss += loss.item() * inputs.size(0)  # Accumulate batch loss

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():  # Do not calculate gradients during validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        # Calculate average loss
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss:.6f} \t\t Validation Loss: {val_loss:.6f}')

    plt.show()
    return train_losses, val_losses

# Plot training and validation losses
def plot_losses(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim(0, 0.05)
    plt.legend()
    plt.savefig(f'{model_name}_losses.png')  # Save the plot as a file
    plt.close()  # Close the plot to free resources

# Test the model
def test_model(model, test_loader):
    model.eval()
    pred_vectors = []
    true_labels = []
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            pred_vectors.append(outputs)
            true_labels.append(labels)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

    # Convert prediction and true labels to suitable format
    pred_vectors = torch.cat(pred_vectors, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    test_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')
    return pred_vectors, true_labels

# Custom VGG model for single-channel input and custom output classes
class CustomVGG(nn.Module):
    def __init__(self, output_features):
        super(CustomVGG, self).__init__()
        # Load pre-trained VGG16 model
        original_model = models.vgg16(pretrained=True)

        # Modify the first layer for single-channel input
        original_model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # Use the feature extraction part of the pre-trained model
        self.features = original_model.features
        self.avgpool = original_model.avgpool

        # Modify the final layer for regression task
        num_ftrs = original_model.classifier[6].in_features
        original_model.classifier[6] = nn.Linear(num_ftrs, output_features)

        # Use the modified classifier
        self.classifier = original_model.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Early Stopping class for regularization
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for saving the model.
            trace_func (function): Function to print messages.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
