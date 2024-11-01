import os
import torch
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
import xml.etree.ElementTree as ET

# Define image transformations (uniform image size)
transform = Compose([
    Resize((224, 224)),  # Resize image to 224x224
    ToTensor(),          # Convert to tensor format
])


def parse_voc_annotation(xml_file, original_size, resized_size=(224, 224)):
    """Parse PASCAL VOC XML file and adjust bounding boxes to match the resized image"""
    orig_w, orig_h = original_size  # Original image size
    resized_w, resized_h = resized_size  # Resized image size

    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        # Adjust bounding box coordinates to match the resized image dimensions
        xmin = int(float(bbox.find("xmin").text)) * resized_w / orig_w
        ymin = int(float(bbox.find("ymin").text)) * resized_h / orig_h
        xmax = int(float(bbox.find("xmax").text)) * resized_w / orig_w
        ymax = int(float(bbox.find("ymax").text)) * resized_h / orig_h

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(1)  # Label unified to 'lego'

    return boxes, labels


class LegoDataset(torch.utils.data.Dataset):
    """Custom Dataset class for loading LEGO data"""

    def __init__(self, dataset_dir, transform=None):
        self.image_dir = os.path.join(dataset_dir, "images")
        self.annotation_dir = os.path.join(dataset_dir, "annotations")
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and corresponding XML annotation file
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        xml_file = self.image_files[idx].replace('.jpg', '.xml')
        annotation_path = os.path.join(self.annotation_dir, xml_file)

        # print(f"Loading image: {img_path}")  # Optional: Print loaded image path

        img = Image.open(img_path).convert("RGB")  # Open image
        orig_size = img.size  # Get original image size

        if self.transform:
            img = self.transform(img)  # Apply image transformations

        # Parse XML file and adjust bounding boxes based on the scaling ratio
        boxes, labels = parse_voc_annotation(annotation_path, orig_size)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return img, target


def custom_collate_fn(batch):
    """Custom collate function to handle targets of different sizes"""
    batch = [b for b in batch if b is not None]  # Skip None samples
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def get_data_loaders(batch_size_train=8, batch_size_val=16, num_workers=4):
    """Create and return DataLoaders for training and validation sets"""
    train_dataset = LegoDataset("./selected_6000/train", transform=transform)
    val_dataset = LegoDataset("./selected_6000/val", transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn
    )

    return train_loader, val_loader


def get_test_loader(batch_size_test=16, num_workers=4):
    """Create and return DataLoader for the test set"""
    test_dataset = LegoDataset("./selected_6000/test", transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size_test, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn
    )

    return test_loader
