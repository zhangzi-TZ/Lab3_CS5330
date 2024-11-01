from Data_Loader_600 import get_data_loaders

if __name__ == '__main__':
    # Get data loaders
    train_loader, _ = get_data_loaders(num_workers=0)  # Set num_workers=0 for testing

    # Test if data loader works correctly
    for images, targets in train_loader:
        print(f"Batch size: {len(images)}")  # Output number of images in batch
        print(f"Image shape: {images[0].shape}")  # Shape of the first image
        print(f"Target example: {targets[0]}")  # Example target annotation
        break  # Exit after the first batch
