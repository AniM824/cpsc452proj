from data import (
    get_cifar10_train_loader, get_cifar10_rotation_test_loader, get_cifar10_test_loader,
    get_cifar100_train_loader, get_cifar100_rotation_test_loader, get_cifar100_test_loader,
    get_emnist_train_loader, get_emnist_rotation_test_loader, get_emnist_test_loader,
    get_isic_train_loader, get_isic_test_loader,
    get_tem_train_loader, get_tem_test_loader
)
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


def train_and_test_model(
    model_class,
    model_name,
    dataset_name,
    num_classes,
    train_loader,
    standard_test_loader,
    rotated_test_loader=None,
    device=device,
    num_epochs=20
):
    """
    Instantiates, trains, and tests a given model class on a specific dataset.

    Args:
        model_class: The class of the model to train.
        model_name (str): A string name for the model (e.g., "Baseline").
        dataset_name (str): Name of the dataset (e.g., "CIFAR100").
        num_classes (int): Number of output classes for the model.
        train_loader: DataLoader for the training data.
        standard_test_loader: DataLoader for the standard (non-rotated) test data.
        rotated_test_loader: DataLoader for the rotated test data (optional).
        device: The device to run the model on ('cuda' or 'cpu').
        num_epochs (int): Number of epochs to train for.

    Returns:
        tuple: (best_standard_test_accuracy, best_rotated_test_accuracy or None) observed during training.
    """
    print(f"\n--- Starting Training for {model_name} Model on {dataset_name} ---")

    results_dir = os.path.join("results", dataset_name.lower())
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    csv_filename = f"{dataset_name.lower()}_{model_name.lower().replace(' ', '_')}_training_metrics.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    header = ['Epoch', 'Loss', 'Accuracy', 'Learning Rate', 'Standard Test Acc']
    if rotated_test_loader:
        header.append('Rotated Test Acc')
    csv_writer.writerow(header)

    model = model_class(num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    learning_rate = 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0
    best_std_test_acc = 0.0
    best_rot_test_acc = 0.0 if rotated_test_loader else None
    base_model_filename = f'best_{dataset_name.lower()}_{model_name.lower().replace(" ", "_")}_model.pth'
    best_results_model_path = os.path.join(models_dir, base_model_filename)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"{dataset_name} {model_name} Epoch {epoch+1}/{num_epochs}", leave=False)

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
        print(f"{dataset_name} {model_name} Epoch {epoch+1:02d} | Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_acc:.2f}% | LR: {current_lr:.6f}")

        # --- Periodic Testing ---
        std_test_acc_epoch = ''
        rot_test_acc_epoch = '' if rotated_test_loader else None

        # Test every 10 epochs or on the last epoch
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            model.eval() # Set model to evaluation mode for testing
            print(f"--- Running periodic testing at Epoch {epoch+1} ---")
            std_test_acc_epoch, _, _ = test_model(
                model, standard_test_loader, device,
                description=f"{dataset_name} {model_name} Standard Test (Epoch {epoch+1})"
            )
            # Only run rotated test if loader is provided
            if rotated_test_loader:
                rot_test_acc_epoch, _, _ = test_model(
                    model, rotated_test_loader, device,
                    description=f"{dataset_name} {model_name} Rotated Test (Epoch {epoch+1})"
                )
                print(f"Epoch {epoch+1} | Standard Test Acc: {std_test_acc_epoch:.2f}% | Rotated Test Acc: {rot_test_acc_epoch:.2f}%")
                best_rot_test_acc = max(best_rot_test_acc, rot_test_acc_epoch)
            else:
                 print(f"Epoch {epoch+1} | Standard Test Acc: {std_test_acc_epoch:.2f}%")

            print(f"-------------------------------------------------")
            best_std_test_acc = max(best_std_test_acc, std_test_acc_epoch)

        row_data = [epoch+1, epoch_loss, epoch_acc, current_lr, std_test_acc_epoch]
        if rotated_test_loader:
            row_data.append(rot_test_acc_epoch)
        csv_writer.writerow(row_data)
        csv_file.flush()

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), best_results_model_path)
            print(f"New best {dataset_name} {model_name} model saved based on train acc: {best_acc:.2f}% to {best_results_model_path}")

    # Close CSV file
    csv_file.close()
    print(f"\nTraining complete for {model_name} on {dataset_name}.")
    print(f"Best Standard Test Accuracy observed during training: {best_std_test_acc:.2f}%")
    if best_rot_test_acc is not None:
        print(f"Best Rotated Test Accuracy observed during training: {best_rot_test_acc:.2f}%")

    return best_std_test_acc, best_rot_test_acc


def test_model(model, test_loader, device, description="Testing"):
    model.eval()
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
    EMNIST_EPOCHS = 30
    CIFAR_10_EPOCHS = 30
    CIFAR_100_EPOCHS = 50
    ISIC_EPOCHS = 50
    TEM_EPOCHS = 50
    BATCH_SIZE = 64

    # CIFAR-10 Config
    ROOT_CIFAR_10 = "./data"
    NUM_CLASSES_CIFAR_10 = 10

    # CIFAR-100 Config
    ROOT_CIFAR_100 = "./data"
    NUM_CLASSES_CIFAR_100 = 100

    # EMNIST Config
    ROOT_EMNIST = "./data"
    SPLIT_EMNIST = 'byclass'
    NUM_CLASSES_EMNIST = 62

    # ISIC Config
    ROOT_ISIC = "./data/isic"
    NUM_CLASSES_ISIC = 9

    # TEM Config
    ROOT_TEM = "./data"
    NUM_CLASSES_TEM = 14

    results = {}

    print("\n===== Starting EMNIST Experiment =====")
    emnist_train_loader = get_emnist_train_loader(root=ROOT_EMNIST, split=SPLIT_EMNIST, batch_size=BATCH_SIZE)
    emnist_test_loader = get_emnist_test_loader(root=ROOT_EMNIST, split=SPLIT_EMNIST, batch_size=BATCH_SIZE)
    emnist_rot_test_loader = get_emnist_rotation_test_loader(root=ROOT_EMNIST, split=SPLIT_EMNIST, batch_size=BATCH_SIZE)

    results["EMNIST"] = {}
    results["EMNIST"]["Baseline"] = train_and_test_model(
        model_class=baseline_small_inception,
        model_name="Baseline",
        dataset_name="EMNIST",
        num_classes=NUM_CLASSES_EMNIST,
        train_loader=emnist_train_loader,
        standard_test_loader=emnist_test_loader,
        rotated_test_loader=emnist_rot_test_loader,
        device=device,
        num_epochs=EMNIST_EPOCHS
    )

    results["EMNIST"]["Equivariant"] = train_and_test_model(
        model_class=EquivariantSmallInception,
        model_name="Equivariant",
        dataset_name="EMNIST",
        num_classes=NUM_CLASSES_EMNIST,
        train_loader=emnist_train_loader,
        standard_test_loader=emnist_test_loader,
        rotated_test_loader=emnist_rot_test_loader,
        device=device,
        num_epochs=EMNIST_EPOCHS
    )
    print("===== EMNIST Experiment Complete =====")

    print("\n===== Starting CIFAR-10 Experiment =====")
    cifar_train_loader = get_cifar10_train_loader(root=ROOT_CIFAR_10, batch_size=BATCH_SIZE)
    cifar_test_loader = get_cifar10_test_loader(root=ROOT_CIFAR_10, batch_size=BATCH_SIZE)
    cifar_rot_test_loader = get_cifar10_rotation_test_loader(root=ROOT_CIFAR_10, batch_size=BATCH_SIZE)

    results["CIFAR10"] = {}
    results["CIFAR10"]["Baseline"] = train_and_test_model(
        model_class=baseline_small_inception,
        model_name="Baseline",
        dataset_name="CIFAR10",
        num_classes=NUM_CLASSES_CIFAR_10,
        train_loader=cifar_train_loader,
        standard_test_loader=cifar_test_loader,
        rotated_test_loader=cifar_rot_test_loader,
        device=device,
        num_epochs=CIFAR_10_EPOCHS
    )

    results["CIFAR10"]["Equivariant"] = train_and_test_model(
        model_class=EquivariantSmallInception,
        model_name="Equivariant",
        dataset_name="CIFAR10",
        num_classes=NUM_CLASSES_CIFAR_10,
        train_loader=cifar_train_loader,
        standard_test_loader=cifar_test_loader,
        rotated_test_loader=cifar_rot_test_loader,
        device=device,
        num_epochs=CIFAR_10_EPOCHS
    )
    print("===== CIFAR-10 Experiment Complete =====")

    print("\n===== Starting CIFAR-100 Experiment =====")
    cifar_train_loader = get_cifar100_train_loader(root=ROOT_CIFAR_100, batch_size=BATCH_SIZE)
    cifar_test_loader = get_cifar100_test_loader(root=ROOT_CIFAR_100, batch_size=BATCH_SIZE)
    cifar_rot_test_loader = get_cifar100_rotation_test_loader(root=ROOT_CIFAR_100, batch_size=BATCH_SIZE)

    results["CIFAR100"] = {}
    results["CIFAR100"]["Baseline"] = train_and_test_model(
        model_class=baseline_small_inception,
        model_name="Baseline",
        dataset_name="CIFAR100",
        num_classes=NUM_CLASSES_CIFAR_100,
        train_loader=cifar_train_loader,
        standard_test_loader=cifar_test_loader,
        rotated_test_loader=cifar_rot_test_loader,
        device=device,
        num_epochs=CIFAR_100_EPOCHS
    )

    results["CIFAR100"]["Equivariant"] = train_and_test_model(
        model_class=EquivariantSmallInception,
        model_name="Equivariant",
        dataset_name="CIFAR100",
        num_classes=NUM_CLASSES_CIFAR_100,
        train_loader=cifar_train_loader,
        standard_test_loader=cifar_test_loader,
        rotated_test_loader=cifar_rot_test_loader,
        device=device,
        num_epochs=CIFAR_100_EPOCHS
    )
    print("===== CIFAR-100 Experiment Complete =====")

    print("\n===== Starting ISIC Experiment =====")
    isic_train_loader = get_isic_train_loader(root=ROOT_ISIC, batch_size=BATCH_SIZE)
    isic_test_loader = get_isic_test_loader(root=ROOT_ISIC, batch_size=BATCH_SIZE)

    results["ISIC"] = {}
    results["ISIC"]["Baseline"] = train_and_test_model(
        model_class=baseline_small_inception,
        model_name="Baseline",
        dataset_name="ISIC",
        num_classes=NUM_CLASSES_ISIC,
        train_loader=isic_train_loader,
        standard_test_loader=isic_test_loader,
        rotated_test_loader=None,
        device=device,
        num_epochs=ISIC_EPOCHS
    )

    results["ISIC"]["Equivariant"] = train_and_test_model(
        model_class=EquivariantSmallInception,
        model_name="Equivariant",
        dataset_name="ISIC",
        num_classes=NUM_CLASSES_ISIC,
        train_loader=isic_train_loader,
        standard_test_loader=isic_test_loader,
        rotated_test_loader=None, 
        device=device,
        num_epochs=ISIC_EPOCHS
    )
    print("===== ISIC Experiment Complete =====")

    print("\n===== Starting TEM 2019 Virus Experiment =====")
    tem_train_loader = get_tem_train_loader(root=ROOT_TEM, batch_size=BATCH_SIZE)
    tem_test_loader = get_tem_test_loader(root=ROOT_TEM, batch_size=BATCH_SIZE)

    results["TEM"] = {}
    results["TEM"]["Baseline"] = train_and_test_model(
        model_class=baseline_small_inception,
        model_name="Baseline",
        dataset_name="TEM",
        num_classes=NUM_CLASSES_TEM,
        train_loader=tem_train_loader,
        standard_test_loader=tem_test_loader,
        rotated_test_loader=None,
        device=device,
        num_epochs=TEM_EPOCHS
    )

    results["TEM"]["Equivariant"] = train_and_test_model(
        model_class=EquivariantSmallInception,
        model_name="Equivariant",
        dataset_name="TEM",
        num_classes=NUM_CLASSES_TEM,
        train_loader=tem_train_loader,
        standard_test_loader=tem_test_loader,
        rotated_test_loader=None, 
        device=device,
        num_epochs=TEM_EPOCHS
    )
    print("===== TEM Experiment Complete =====")

