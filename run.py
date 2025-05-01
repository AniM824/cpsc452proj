from data import get_cifar100_train_loader, get_cifar100_rotation_test_loader, get_cifar100_test_loader
from baseline import baseline_small_inception
from equivariant import EquivariantSmallInception
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def train_and_test_model(model_class, model_name, num_classes, train_loader, device, num_epochs=20):
    """
    Instantiates, trains, and tests a given model class.

    Args:
        model_class: The class of the model to train (e.g., baseline_small_inception).
        model_name (str): A string name for the model (e.g., "Baseline").
        num_classes (int): Number of output classes for the model.
        train_loader: DataLoader for the training data.
        device: The device to run the model on ('cuda' or 'cpu').
        num_epochs (int): Number of epochs to train for.

    Returns:
        tuple: (standard_test_accuracy, rotated_test_accuracy)
    """
    print(f"\n--- Starting Training for {model_name} Model ---")

    # Create results directory
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    
    # Create CSV file for training metrics
    csv_path = os.path.join("results", f"{model_name.lower().replace(' ', '_')}_training_metrics.csv")
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Loss', 'Accuracy', 'Learning Rate'])
    
    model = model_class(num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    if model_name == "Equivariant":
        conv_params = []
        fc_params = []
        
        for name, param in model.named_parameters():
            if 'fully_connected' in name:
                fc_params.append(param)
            else:
                conv_params.append(param)
                
        optimizer = optim.AdamW([
            {'params': conv_params, 'lr': 3e-3},
            {'params': fc_params, 'lr': 1e-3}
        ], weight_decay=1e-4)
    else:
        learning_rate = 1e-3
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0
    os.makedirs("saved_models", exist_ok=True)
    best_model_path = os.path.join("saved_models", f'best_{model_name.lower().replace(" ", "_")}_model.pth')
    # Also save in results directory
    best_results_model_path = os.path.join("results/models", f'best_{model_name.lower().replace(" ", "_")}_model.pth')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{num_epochs}", leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            current_loss = running_loss / total
            current_acc = 100. * correct / total
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

        scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        current_lr = scheduler.get_last_lr()[0]
        print(f"{model_name} Epoch {epoch+1:02d} | Final Loss: {epoch_loss:.4f} | Final Accuracy: {epoch_acc:.2f}% | LR: {current_lr:.6f}")
        
        # Write metrics to CSV
        csv_writer.writerow([epoch+1, epoch_loss, epoch_acc, current_lr])
        csv_file.flush()

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), best_model_path)
            torch.save(model.state_dict(), best_results_model_path)
            print(f"New best {model_name} model saved with accuracy: {best_acc:.2f}% to {best_model_path}")

    # Close CSV file
    csv_file.close()
    print(f"\nTraining complete for {model_name}. Loading best model for testing...")

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    print(f"\nTesting {model_name} on standard CIFAR-100 test set...")
    standard_test_loader = get_cifar100_test_loader(root="./data")
    standard_test_acc, _, _ = test_model(model, standard_test_loader, device, description=f"{model_name} Standard Test")

    print(f"\nTesting {model_name} on rotated CIFAR-100 test set...")
    rotated_test_loader = get_cifar100_rotation_test_loader(root="./data")
    rotated_test_acc, _, _ = test_model(model, rotated_test_loader, device, description=f"{model_name} Rotated Test")

    print(f"\n--- {model_name} Model Results ---")
    print(f"Final Standard Test Accuracy: {standard_test_acc:.2f}%")
    print(f"Final Rotated Test Accuracy: {rotated_test_acc:.2f}%")
    print(f"----------------------------------")
    
    # Write test results to file
    results_file_path = os.path.join("results", f"{model_name.lower().replace(' ', '_')}_test_results.txt")
    with open(results_file_path, 'w') as f:
        f.write(f"--- {model_name} Model Results ---\n")
        f.write(f"Final Standard Test Accuracy: {standard_test_acc:.2f}%\n")
        f.write(f"Final Rotated Test Accuracy: {rotated_test_acc:.2f}%\n")
        f.write(f"----------------------------------\n")
    
    print(f"Test results saved to {results_file_path}")

    return standard_test_acc, rotated_test_acc


def test_model(model, test_loader, device, description="Testing"):
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        test_progress = tqdm(test_loader, desc=description)
        for data in test_progress:
            if len(data) == 3:
                inputs, labels, _ = data
            else:
                inputs, labels = data
                
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            current_acc = 100. * correct / total
            test_progress.set_postfix(acc=f"{current_acc:.2f}%")

    test_acc = 100. * correct / total

    return test_acc, all_predictions, all_labels



if __name__ == "__main__":
    train_loader = get_cifar100_train_loader(root="./data")

    NUM_CLASSES = 100
    NUM_EPOCHS = 50

    baseline_std_acc, baseline_rot_acc = train_and_test_model(
        model_class=baseline_small_inception,
        model_name="Baseline",
        num_classes=NUM_CLASSES,
        train_loader=train_loader,
        device=device,
        num_epochs=NUM_EPOCHS
    )

    equivariant_std_acc, equivariant_rot_acc = train_and_test_model(
        model_class=EquivariantSmallInception,
        model_name="Equivariant",
        num_classes=NUM_CLASSES,
        train_loader=train_loader,
        device=device,
        num_epochs=NUM_EPOCHS
    )

    print("\n===== Overall Results =====")
    print(f"Baseline Model:")
    print(f"  Standard Test Accuracy: {baseline_std_acc:.2f}%")
    print(f"  Rotated Test Accuracy:  {baseline_rot_acc:.2f}%")
    print(f"\nEquivariant Model:")
    print(f"  Standard Test Accuracy: {equivariant_std_acc:.2f}%")
    print(f"  Rotated Test Accuracy:  {equivariant_rot_acc:.2f}%")
    print("==========================")
    
    # Write overall results to file
    overall_results_path = os.path.join("results", "overall_results.txt")
    with open(overall_results_path, 'w') as f:
        f.write("===== Overall Results =====\n")
        f.write(f"Baseline Model:\n")
        f.write(f"  Standard Test Accuracy: {baseline_std_acc:.2f}%\n")
        f.write(f"  Rotated Test Accuracy:  {baseline_rot_acc:.2f}%\n")
        f.write(f"\nEquivariant Model:\n")
        f.write(f"  Standard Test Accuracy: {equivariant_std_acc:.2f}%\n")
        f.write(f"  Rotated Test Accuracy:  {equivariant_rot_acc:.2f}%\n")
        f.write("==========================\n")
    
    print(f"Overall results saved to {overall_results_path}")


