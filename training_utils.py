import torch
from torch.utils.data import DataLoader, TensorDataset

def train_neural_net(model, data_loader, loss_fn, optim_func, max_epochs=100, early_stop_patience=3):
    model.train()
    min_loss = float('inf')  
    stagnant_epochs = 0  
    loss_history = []
    accuracy_history = []

    for epoch in range(max_epochs):
        epoch_loss = 0.0
        correct_count = 0  
        total_count = 0  

        for batch_images, batch_labels in data_loader:
            optim_func.zero_grad()
            predictions = model(batch_images)
            loss_value = loss_fn(predictions, batch_labels)
            loss_value.backward()
            optim_func.step()
            
            epoch_loss += loss_value.item()
            _, max_indices = predictions.max(1)
            correct_count += max_indices.eq(batch_labels).sum().item()
            total_count += batch_labels.size(0)
        
        avg_epoch_loss = epoch_loss / len(data_loader)
        loss_history.append(avg_epoch_loss)
        epoch_accuracy = 100 * correct_count / total_count
        accuracy_history.append(epoch_accuracy)
        
        print(f'Epoch {epoch + 1}/{max_epochs}, Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

        if avg_epoch_loss < min_loss:
            min_loss = avg_epoch_loss
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1

        if stagnant_epochs >= early_stop_patience and epoch > 10:
            print(f'Early stopping triggered at epoch {epoch + 1} after {early_stop_patience} stagnant epochs.')
            break
            
    return loss_history, accuracy_history


def test_neural_net(model, data_loader, loss_fn, device='cpu'):
    model.eval()
    total = 0
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += loss_fn(outputs, target).item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = test_loss / len(data_loader)
    accuracy = 100 * correct / total
    print(f"Test set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)")

    return avg_loss, accuracy

def save_model(model, path):
    """Saves the model state dictionary to the specified path."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model_class, path, num_labels):
    """Loads the model state dictionary from the specified path and returns the model."""
    model = model_class(num_labels)  # Create a new instance of the model
    model.load_state_dict(torch.load(path))  # Load the state dictionary
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded successfully from {path}")
    return model


def create_combined_dataloader(data_loader, train_attack, batch_size=4):
    """
    Create a DataLoader containing both clean and adversarial examples.

    Args:
        data_loader (DataLoader): DataLoader for the original clean training data.
        train_attack: An adversarial attack instance (e.g., FastGradientMethod).
        batch_size (int): Batch size for the combined DataLoader.

    Returns:
        DataLoader: A new DataLoader containing clean and adversarial examples.
    """
    all_clean_images = []
    all_adv_images = []
    all_labels = []

    # Iterate through the original DataLoader
    for images, labels in data_loader:
        # Convert images to NumPy for ART compatibility
        images_np = images.numpy()

        # Generate adversarial examples
        adv_images_np = train_attack.generate(x=images_np)

        # Convert adversarial examples back to tensors
        adv_images = torch.tensor(adv_images_np, dtype=torch.float32)

        # Collect clean and adversarial examples
        all_clean_images.append(images)
        all_adv_images.append(adv_images)
        all_labels.append(labels)

    # Concatenate all tensors
    all_clean_images = torch.cat(all_clean_images, dim=0)
    all_adv_images = torch.cat(all_adv_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Combine clean and adversarial examples
    combined_images = torch.cat((all_clean_images, all_adv_images), dim=0)
    combined_labels = torch.cat((all_labels, all_labels), dim=0)  # Labels remain the same for both

    # Create a new DataLoader
    combined_dataset = TensorDataset(combined_images, combined_labels)
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    return combined_loader


def train_with_adversarial_examples(model, data_loader, optimizer, loss_fn, train_attack, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_adv_correct = 0
        total_samples = 0

        for images, labels in data_loader:
            # Generate adversarial examples
            images_np = images.numpy()  # Convert images to numpy array for ART
            images_adv_np = train_attack.generate(x=images_np)
            images_adv = torch.tensor(images_adv_np).float()  # Convert back to tensor

            optimizer.zero_grad()

            # Forward pass with clean images
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()  # Backward pass with clean images

            # Forward pass with adversarial examples
            outputs_adv = model(images_adv)
            loss_adv = loss_fn(outputs_adv, labels)
            loss_adv.backward()  # Backward pass with adversarial examples

            optimizer.step()

            total_loss += loss.item() + loss_adv.item()

            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()

            _, predicted_adv = torch.max(outputs_adv.data, 1)
            total_adv_correct += (predicted_adv == labels).sum().item()

            total_samples += labels.size(0)

        # Print statistics
        avg_loss = total_loss / (2 * len(data_loader))  # Factor of 2 because of clean and adv examples
        accuracy = 100 * total_correct / total_samples
        adv_accuracy = 100 * total_adv_correct / total_samples

        print(f'Epoch {epoch+1}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Adversarial Accuracy: {adv_accuracy:.2f}%')


def evaluate_model_on_clean_and_adversarial(model, data_loader, loss_fn, attack_method, device='cpu'):
    model.eval()  # Ensure the model is in evaluation mode
    device = torch.device(device)
    model.to(device)

    total_clean_correct = 0
    total_adv_correct = 0
    total_samples = 0
    total_clean_loss = 0
    total_adv_loss = 0

    with torch.no_grad():  # Typically, no grad is needed for evaluation
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Clean images forward pass
            clean_outputs = model(images)
            clean_loss = loss_fn(clean_outputs, labels)
            total_clean_loss += clean_loss.item()

            _, clean_predictions = torch.max(clean_outputs.data, 1)
            total_clean_correct += (clean_predictions == labels).sum().item()

    # Grad is required for generating adversarial examples
    model.train()  # Switch to training mode for generating adversarial examples
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True  # Ensure gradients are tracked

        # Generate adversarial examples
        images_np = images.cpu().detach().numpy()  # Detach and move to cpu
        images_adv_np = attack_method.generate(x=images_np)
        images_adv = torch.tensor(images_adv_np).float().to(device)  # Convert back to tensor

        model.eval()  # Switch back to eval mode for prediction
        with torch.no_grad():  # Disable gradient calculation for evaluation
            adv_outputs = model(images_adv)
            adv_loss = loss_fn(adv_outputs, labels)
            total_adv_loss += adv_loss.item()

            _, adv_predictions = torch.max(adv_outputs.data, 1)
            total_adv_correct += (adv_predictions == labels).sum().item()

            total_samples += labels.size(0)

    model.eval()  # Ensure the model is back in evaluation mode after the loop

    # Calculate average loss and accuracy
    avg_clean_loss = total_clean_loss / len(data_loader)
    avg_adv_loss = total_adv_loss / len(data_loader)
    clean_accuracy = 100 * total_clean_correct / total_samples
    adv_accuracy = 100 * total_adv_correct / total_samples

    print(f'Clean Data - Loss: {avg_clean_loss:.4f}, Accuracy: {clean_accuracy:.2f}%')
    print(f'Adversarial Data - Loss: {avg_adv_loss:.4f}, Accuracy: {adv_accuracy:.2f}%')

    return avg_clean_loss, clean_accuracy, avg_adv_loss, adv_accuracy

