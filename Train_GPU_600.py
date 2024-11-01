import torch
from torch.optim import SGD
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from Data_Loader_600 import get_data_loaders
from get_model import get_model
import matplotlib.pyplot as plt


def evaluate_model(model, val_loader, device):
    """
    Evaluate the model on the validation set using mAP@0.5.

    Args:
        model (torch.nn.Module): The trained model.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): The device to perform computations on.

    Returns:
        float: The mean Average Precision at IoU threshold 0.5.
    """
    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5])
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            metric.update(outputs, targets)

    mAP_results = metric.compute()
    model.train()  # Switch back to training mode
    return mAP_results['map_50'].item()


def train_model(model, train_loader, val_loader, device, num_epochs=10):
    """
    Train the Faster R-CNN model with validation and plot the loss and mAP curves.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): The device to perform computations on.
        num_epochs (int, optional): Number of training epochs. Defaults to 10.
    """
    model.train()  # Switch to training mode
    optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    best_map = 0.0  # To save the best validation mAP

    epoch_losses = []  # List to store average loss per epoch
    epoch_map = []      # List to store mAP per epoch

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        print(f"\nStarting epoch {epoch + 1}/{num_epochs}...")  # Print epoch start

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            num_batches += 1

            # Print loss for each batch
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {losses.item():.4f}")

        # Calculate average loss for the epoch
        average_loss = total_loss / num_batches
        epoch_losses.append(average_loss)
        print(f"Epoch {epoch + 1} completed. Average Loss: {average_loss:.4f}")

        # Evaluate on validation set
        val_map = evaluate_model(model, val_loader, device)
        epoch_map.append(val_map)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation mAP@0.5: {val_map:.4f}")

        # Save the best model based on validation mAP
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), "best_model_gpu_600.pth")
            print("Best model saved.")

    print("\nTraining complete.")
    print(f"Best Validation mAP@0.5: {best_map:.4f}")

    # Plot the training loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('gpu_loss_curve.png')  # Save the plot as an image file
    plt.show()

    # Plot the validation mAP curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), epoch_map, marker='o', color='green', label='Validation mAP@0.5')
    plt.xlabel('Epoch')
    plt.ylabel('mAP@0.5')
    plt.title('Validation mAP@0.5 Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('gpu_map_curve.png')  # Save the plot as an image file
    plt.show()


if __name__ == '__main__':
    # Set device to GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("No GPU found, using CPU")

    train_loader, val_loader = get_data_loaders(batch_size_train=4, batch_size_val=8, num_workers=4)

    # Load model and move to device
    model = get_model()
    model.to(device)

    # Train model
    train_model(model, train_loader, val_loader, device, num_epochs=2)
