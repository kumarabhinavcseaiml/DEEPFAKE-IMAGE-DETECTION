# Deepfake Lens

## Student Details

- **Name:** Kumar Abhinav
- **College:** Ghani Khan Choudhury Institute of Engineering and Technology
- **Department:** CSE (AI & ML)

Deepfake Lens is an educational real-versus-fake face image classification project. It extracts Histogram of Oriented Gradients (HOG) features from images, compares three classical machine-learning models, saves the best model, and provides an interactive Streamlit dashboard for image prediction.

The project currently includes 5,215 readable images: 2,215 real and 3,000 fake images.

## Features

- Trains Logistic Regression, Random Forest, and SVM classifiers
- Uses HOG feature extraction on 64 x 64 grayscale images
- Creates stratified 60% training, 20% validation, and 20% test splits
- Selects the best model using validation F1-score
- Generates confusion matrices, ROC curves, comparison charts, and classification reports
- Provides image upload and prediction through a Streamlit interface
- Saves the selected model as `models/best_model.joblib`

## Project Structure

```text
.
|-- app.py                     # Streamlit web application
|-- train.py                   # Dataset processing, training, and evaluation
|-- requirements.txt           # Python dependencies
|-- data/
|   `-- raw/
|       |-- real/              # Real face images
|       `-- fake/              # Fake face images
|-- models/
|   `-- best_model.joblib      # Saved model bundle
`-- reports/
    |-- dataset_summary.json
    |-- dataset_manifest.csv
    |-- model_comparison.csv
    |-- model_comparison.png
    |-- roc_curves.png
    |-- class_distribution.png
    |-- sample_images.png
    `-- *_confusion_matrix.png
```

## Dataset

Place images in the following directories:

```text
data/raw/real/
data/raw/fake/
```

The supported formats are JPG, JPEG, PNG, BMP, and WebP. Subfolders inside each class directory are scanned recursively. At least five readable images from each class are required.

## Installation

Python 3.13 is used by the included project environment on Windows.

### Windows

```powershell
cd "C:\Users\razch\OneDrive\Desktop\ml project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If a virtual environment named `.venv313` is already present, use:

```powershell
.\\.venv313\Scripts\python.exe -m pip install -r requirements.txt
```

## Train the Models

Run the training script from the project root:

```powershell
python train.py
```

With the included environment:

```powershell
.\\.venv313\Scripts\python.exe train.py
```

Training loads and validates the images, extracts HOG features, trains all three models, evaluates them on the test set, and writes results to `reports/`. It also saves the model with the highest validation FAKE-class F1-score to `models/best_model.joblib`.

## Run the Streamlit App

```powershell
streamlit run app.py
```

With the included environment:

```powershell
.\\.venv313\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. The dashboard includes:

- **Overview**: dataset and project summary
- **Detect an image**: upload an image and generate a REAL/FAKE prediction
- **Model performance**: compare algorithms and view confusion matrices
- **About the project**: workflow and objectives

## Saved Results

The current model bundle selects **SVM** as the saved best model.

The held-out test results saved in `reports/model_comparison.csv` are:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 65.77% | 70.84% | 68.83% | 69.82% |
| Random Forest | 68.84% | 71.25% | 76.83% | 73.94% |
| SVM | 74.59% | 76.05% | 81.50% | 78.68% |

These metrics treat **FAKE** as the positive class. Scores depend on the supplied dataset and random split and should not be interpreted as a guarantee of real-world deepfake detection accuracy.

## How Prediction Works

1. The uploaded image is converted to RGB.
2. OpenCV converts it to grayscale.
3. The image is resized to 64 x 64 pixels.
4. HOG features are extracted using the same settings as training.
5. The saved classifier predicts REAL or FAKE.
6. Streamlit displays the selected class and available probability estimates.

## Limitations

- This is a classical ML prototype, not a production deepfake detection system.
- Results may degrade on unseen resolutions, compression levels, camera types, or editing methods.
- HOG features can miss high-level semantic clues learned by modern deep networks.
- Confidence scores are model estimates, not proof of authenticity.

