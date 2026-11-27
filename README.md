# CIFAR-10 Image Classifier using a Convolutional Neural Network (CNN)

A deep learning project that classifies 32x32 colour images into 10 categories
(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) using a
custom CNN built with TensorFlow/Keras, with a Streamlit web app for live demo.

## Tech Stack
Python, TensorFlow/Keras, NumPy, scikit-learn, Matplotlib, Pillow, Streamlit

## Project Structure
```
cifar10-cnn-classifier/
├── app.py              # Streamlit web app (upload image -> prediction)
├── requirements.txt
├── src/
│   ├── config.py       # constants & paths
│   ├── data.py         # load + split dataset
│   ├── model.py        # CNN architecture
│   ├── train.py        # training loop + callbacks + curves
│   ├── evaluate.py     # test metrics + confusion matrix
│   └── predict.py      # single-image inference
├── models/             # saved best_model.keras
├── outputs/            # training_curves.png, confusion_matrix.png
└── sample_images/      # your own test images
```

## Pipeline
Data (CIFAR-10) -> Split (45k train / 5k val / 10k test) -> Rescale + Augment
-> CNN (3 conv blocks) -> Softmax (10 classes) -> Evaluate -> Streamlit app

## Model Architecture
| Stage | Details |
|-------|---------|
| Preprocessing | Rescale 1/255, RandomFlip, RandomRotation, RandomZoom |
| Block 1 | 2x [Conv 32 + BN + ReLU] -> MaxPool -> Dropout 0.2 |
| Block 2 | 2x [Conv 64 + BN + ReLU] -> MaxPool -> Dropout 0.3 |
| Block 3 | 2x [Conv 128 + BN + ReLU] -> MaxPool -> Dropout 0.4 |
| Head | GlobalAvgPool -> Dense 128 -> Dropout 0.5 -> Dense 10 (softmax) |

Optimizer: Adam (lr 1e-3) | Loss: sparse categorical cross-entropy
Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

## How to Run

python -m venv venv
python -m venv venv     
pip install -r requirements.txt

python -m src.train             
python -m src.evaluate          
python -m src.predict sample_images/cat.jpg
streamlit run app.py            


## Results
Fill in after training (typically ~85-90% test accuracy for this architecture):
- Test accuracy: **83.4%**
- See `outputs/training_curves.png` and `outputs/confusion_matrix.png`

## Possible Improvements
Transfer learning (MobileNetV2), learning-rate schedules, Grad-CAM
visualisation, Docker + deployment on Hugging Face Spaces.
