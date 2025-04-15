# Age Detection Model

This project implements a deep learning model for age detection using the IMDB-WIKI dataset. The model is based on a fine-tuned VGG16 architecture.

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Dataset

The model is designed to work with the IMDB-WIKI dataset. You need to:
1. Download the IMDB-WIKI dataset
2. Update the `data_path` variable in `age_detection.py` to point to your dataset location
3. Ensure the dataset structure matches the expected format for age extraction

## Training the Model

To train the model, simply run:
```bash
python age_detection.py
```

The script will:
1. Load and preprocess the dataset
2. Create and compile the model
3. Train the model with data augmentation
4. Save the trained model as 'age_detection_model.h5'
5. Generate training history plots

## Model Architecture

The model uses:
- VGG16 as the base model (pre-trained on ImageNet)
- Global Average Pooling layer
- Dense layers with ReLU activation
- Dropout for regularization
- Linear output layer for age prediction

## Training Parameters

- Batch size: 32
- Learning rate: 0.001
- Epochs: 10 (adjustable)
- Loss function: Mean Squared Error
- Metric: Mean Absolute Error

## Output

After training, you'll get:
- Trained model: 'age_detection_model.h5'
- Training history plot: 'training_history.png' 