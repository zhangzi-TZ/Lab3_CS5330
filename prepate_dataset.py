import os
import random
import shutil
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import argparse


def unify_annotations(original_annotations_dir, unified_annotations_dir):
    """
    Convert all object class labels in the annotation files to 'lego'.
    """
    os.makedirs(unified_annotations_dir, exist_ok=True)

    def convert_annotation_to_lego(xml_file, output_dir):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Iterate through all objects and set label to 'lego'
            for obj in root.findall("object"):
                obj.find("name").text = "lego"

            new_file_path = os.path.join(output_dir, os.path.basename(xml_file))
            tree.write(new_file_path)
            print(f"Processed annotation: {new_file_path}")

        except ET.ParseError as e:
            print(f"Error parsing {xml_file}: {e}")

    # Process each XML file
    for xml_file in os.listdir(original_annotations_dir):
        if xml_file.endswith(".xml"):
            convert_annotation_to_lego(os.path.join(original_annotations_dir, xml_file), unified_annotations_dir)


def is_annotation_valid(image_path, xml_path, resized_size=(224, 224)):
    """
    Validate that the bounding boxes in the XML file are within the image dimensions.
    """
    try:
        img = Image.open(image_path)
        width, height = img.size

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))

            if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                print(f"Invalid bbox in {xml_path}: ({xmin}, {ymin}, {xmax}, {ymax})")
                return False
        return True
    except Exception as e:
        print(f"Error validating {image_path} or {xml_path}: {e}")
        return False


def collect_valid_images(image_dir, annotation_dir):
    """
    Collect all valid images that have corresponding valid annotations.
    """
    valid_images = []
    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith('.jpg') or img_file.lower().endswith('.png'):
            xml_file = img_file.rsplit('.', 1)[0] + '.xml'
            xml_path = os.path.join(annotation_dir, xml_file)

            if os.path.exists(xml_path):
                image_path = os.path.join(image_dir, img_file)
                if is_annotation_valid(image_path, xml_path):
                    valid_images.append(img_file)
            else:
                print(f"Annotation file does not exist for image: {img_file}")

    print(f"Total valid images found: {len(valid_images)}")
    return valid_images


def split_dataset(selected_images, base_dir, num_images, image_dir, annotation_dir):
    """
    Split the selected images into training, validation, and testing sets.
    """
    if len(selected_images) < num_images:
        raise ValueError(f"Not enough valid images to select {num_images} samples.")

    sampled_images = random.sample(selected_images, num_images)
    train_split = int(0.70 * num_images)
    val_split = int(0.85 * num_images)

    train_images = sampled_images[:train_split]
    val_images = sampled_images[train_split:val_split]
    test_images = sampled_images[val_split:]

    # Define directories
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    # Create directory structure
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(directory, "images"), exist_ok=True)
        os.makedirs(os.path.join(directory, "annotations"), exist_ok=True)

    # Function to copy files
    def copy_files(image_list, src_img_dir, src_ann_dir, dest_dir):
        for img_file in image_list:
            # Copy image
            shutil.copy(
                os.path.join(src_img_dir, img_file),
                os.path.join(dest_dir, "images", img_file)
            )

            # Copy annotation
            xml_file = img_file.rsplit('.', 1)[0] + '.xml'
            shutil.copy(
                os.path.join(src_ann_dir, xml_file),
                os.path.join(dest_dir, "annotations", xml_file)
            )

    # Copy files to respective directories
    copy_files(train_images, image_dir, annotation_dir, train_dir)
    copy_files(val_images, image_dir, annotation_dir, val_dir)
    copy_files(test_images, image_dir, annotation_dir, test_dir)

    print(
        f"Dataset '{base_dir}' split into Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")


def count_images(base_dir):
    """
    Count and print the number of images in each split.
    """
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split, "images")
        count = len([file for file in os.listdir(split_dir) if file.lower().endswith(('.jpg', '.png'))])
        print(f"{split.capitalize()} set has {count} images.")


def prepare_dataset(original_annotations_dir, unified_annotations_dir, image_dir, selected_sizes):
    """
    Prepare datasets by unifying annotations, selecting images, splitting, copying, and counting.
    """
    # Step 1: Unify annotations
    print("Unifying annotations...")
    unify_annotations(original_annotations_dir, unified_annotations_dir)

    # Step 2: Collect valid images
    print("\nCollecting valid images...")
    valid_images = collect_valid_images(image_dir, unified_annotations_dir)

    # Step 3: Split datasets for each selected size
    for size in selected_sizes:
        print(f"\nPreparing dataset with {size} images...")
        base_dir = f"./selected_{size}"
        split_dataset(valid_images, base_dir, size, image_dir, unified_annotations_dir)

        # Step 4: Count images in each split
        count_images(base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare LEGO datasets by unifying annotations, selecting, splitting, and counting images.")
    parser.add_argument('--original_annotations_dir', type=str, default="./annotations",
                        help="Path to the original annotations directory.")
    parser.add_argument('--unified_annotations_dir', type=str, default="./annotations_unified",
                        help="Path to save unified annotations.")
    parser.add_argument('--image_dir', type=str, default="./images", help="Path to the images directory.")
    parser.add_argument('--selected_sizes', type=int, nargs='+', default=[600, 6000],
                        help="List of dataset sizes to prepare (e.g., 600 6000).")

    args = parser.parse_args()

    # Set seed for reproducibility
    random.seed(42)

    prepare_dataset(
        original_annotations_dir=args.original_annotations_dir,
        unified_annotations_dir=args.unified_annotations_dir,
        image_dir=args.image_dir,
        selected_sizes=args.selected_sizes
    )
