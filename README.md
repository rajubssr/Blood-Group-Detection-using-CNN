# Blood Group Detection from Fingerprint Images

A deep learning project that predicts blood group from fingerprint images using a hybrid CNN-LSTM architecture.

## Features
- Hybrid CNN-LSTM model for spatial + sequential feature learning
- OpenCV preprocessing pipeline: noise reduction, histogram equalisation, Laplacian edge enhancement
- Predicts all 8 blood groups: A+, A-, B+, B-, AB+, AB-, O+, O-
- Flask web app for real-time predictions from uploaded images
- Evaluation with accuracy, precision, recall, F1-score

## Tech Stack
- Python, Flask
- TensorFlow / Keras
- OpenCV, NumPy
- scikit-learn (evaluation)

## Project Structure
```
blood_group_detection/
├── model.py            # CNN-LSTM architecture
├── train.py            # Training + evaluation script
├── app.py              # Flask web server
├── templates/
│   └── index.html      # Upload UI
├── dataset/            # Place your dataset here (one folder per blood group)
└── requirements.txt
```

## Dataset Format
```
dataset/
├── A+/   (fingerprint images)
├── A-/
├── B+/
...
└── O-/
```

## Setup & Run

```bash
pip install -r requirements.txt

# Train the model
python train.py

# Run the app
python app.py
```

Open http://localhost:5000, upload a fingerprint image, and get the predicted blood group.

## Model Architecture
- 3x Conv2D blocks with BatchNorm + MaxPooling
- Reshape → LSTM(128)
- Dense(64) → Dropout → Softmax output (8 classes)
- Loss: Categorical Crossentropy | Optimizer: Adam
