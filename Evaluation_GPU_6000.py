import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from get_model import get_model
from Data_Loader_6000 import get_test_loader

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("No GPU found, using CPU")


def evaluate_model(model, test_loader, device):
    """Evaluate the model on the test set using mAP@0.5"""
    model.eval()  # Set to evaluation mode
    metric = MeanAveragePrecision(iou_thresholds=[0.5])  # Initialize metric with IoU threshold 0.5

    with torch.no_grad():  # Disable gradient computation
        for images, targets in test_loader:
            images = [img.to(device) for img in images]  # Move images to device
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Move targets to device

            outputs = model(images)  # Get model predictions

            # Update metric with predictions and targets
            metric.update(outputs, targets)

    # Compute mAP
    mAP_results = metric.compute()
    print(f"Test mAP@0.5: {mAP_results['map_50']:.4f}")


if __name__ == '__main__':
    # Get test data loader
    test_loader = get_test_loader(batch_size_test=16, num_workers=4)

    # Load model and move to device
    model = get_model()
    model.to(device)

    # Load trained model weights
    model.load_state_dict(torch.load("best_model_gpu.pth", map_location=device))

    # Evaluate model
    evaluate_model(model, test_loader, device)
