# Lab3_CS5330

## Requirements

- **Python 3.12** 
- **PyTorch**
- **torchvision**
- **PIL**
- **matplotlib**
- **torchmetrics**

## Workflow
- **prepare_dataset.py**
- **Train_CPU/GPU_600/6000.py**
- **Evaluation_CPU/GPU_600/6000.py**

## File Structure

### 1. Data Preparation

- **prepare_dataset.py**: Data preparation script that cleans and formats the raw dataset to make it suitable for model training.

### 2. Data Loading

- **Data_Loader_600.py**: Data loader script for the dataset with 600 images.
- **Data_Loader_6000.py**: Data loader script for the dataset with 6000 images.
- **Test_Dataloader.py**: Script for testing the data loader to ensure it works correctly.

### 3. Model Definition

- **get_model.py**: Defines the Faster R-CNN model with a ResNet-50 backbone, including loading pre-trained weights.

### 4. Model Training

Separate training scripts are provided for different dataset sizes (600 and 6000) and devices (CPU and GPU):

- **Train_CPU_600.py**: Training script for 600 images on CPU.
- **Train_CPU_6000.py**: Training script for 6000 images on CPU.
- **Train_GPU_600.py**: Training script for 600 images on GPU.
- **Train_GPU_6000.py**: Training script for 6000 images on GPU.

### 5. Model Evaluation

Evaluation scripts are provided to calculate the model's performance (e.g., mAP) on the validation set for different dataset sizes and devices:

- **Evaluation_CPU_600.py**: Evaluation script for the model trained on 600 images on CPU.
- **Evaluation_CPU_6000.py**: Evaluation script for the model trained on 6000 images on CPU.
- **Evaluation_GPU_600.py**: Evaluation script for the model trained on 600 images on GPU.
- **Evaluation_GPU_6000.py**: Evaluation script for the model trained on 6000 images on GPU.

### 6. Visualization

- **loss_curve.png**: Visualization of the loss curve over the training process.
- **map_curve.png**: Visualization of the mAP (mean Average Precision) curve over the training process on the validation set.

