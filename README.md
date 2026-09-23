# 🧠 AI Image Classifier

A CNN-based image classification application trained on the CIFAR-10 dataset and deployed with Streamlit.

## 🚀 Live Demo

https://ai-image-classifier-4ipjdqlnh8e4ecufskxyv3.streamlit.app/

## 📌 Project Overview

This project is an AI image classification system that uses a Convolutional Neural Network (CNN) to classify images into 10 CIFAR-10 categories.

The application provides a simple web interface where users can upload an image and receive:

* Predicted image class
* Prediction confidence
* Probability for each supported class
* Technical information about the processed image

## 🎯 Supported Classes

The model can classify images into these 10 categories:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## 🧠 Model

The project uses a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset.

### Input

* Image size: `32 × 32 × 3`
* RGB color images
* Pixel normalization from `0–255` to `0–1`

### Dataset

CIFAR-10 contains 60,000 labeled color images across 10 classes.

The dataset was divided into:

* Training: 45,000 images
* Validation: 5,000 images
* Testing: 10,000 images

## 📊 Model Performance

The final Version 1 model achieved:

| Metric        |     Result |
| ------------- | ---------: |
| Test Accuracy | **68.82%** |
| Test Loss     | **0.9148** |
| Test Images   | **10,000** |

### Class-wise Accuracy

| Class      | Accuracy |
| ---------- | -------: |
| Airplane   |   75.00% |
| Automobile |   86.30% |
| Bird       |   45.60% |
| Cat        |   48.80% |
| Deer       |   58.60% |
| Dog        |   57.40% |
| Frog       |   86.10% |
| Horse      |   82.20% |
| Ship       |   64.30% |
| Truck      |   83.90% |

## ✨ Features

* 🧠 CNN-based image classification
* 📤 Image upload through Streamlit
* 🖼️ Automatic RGB conversion
* 📐 Automatic resizing to `32 × 32`
* 🔢 Pixel normalization
* 🎯 Predicted class display
* 📊 Confidence score
* 📈 Class probability visualization
* 🔧 Technical preprocessing details
* 🌐 Streamlit Community Cloud deployment

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pillow
* Scikit-learn
* Streamlit
* Git
* GitHub

## 📂 Project Structure

```text
AI-image-classifier/
│
├── app.py
├── cnn_model.keras
├── requirements.txt
├── README.md
├── .gitignore
│
└── src/
    ├── data/
    │   ├── augmentation.py
    │   ├── dataset_loader.py
    │   ├── preprocess.py
    │   ├── train_pipeline.py
    │   ├── validation_test_pipeline.py
    │   ├── verify_augmentation.py
    │   ├── verify_preprocessing.py
    │   └── visualize_dataset.py
    │
    └── models/
        ├── callbacks.py
        ├── cnn_model.py
        ├── evaluate_model.py
        ├── predict.py
        └── train_model.py
```

## ⚙️ Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/abdullah8090butt-max/AI-image-classifier.git
```

### 2. Open the project

```bash
cd AI-image-classifier
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 🔄 How the Application Works

```text
User uploads image
        ↓
Convert image to RGB
        ↓
Resize image to 32 × 32
        ↓
Normalize pixel values
        ↓
CNN model prediction
        ↓
Find highest probability class
        ↓
Display class + confidence
        ↓
Display all class probabilities
```

## 🧪 Evaluation

The model was evaluated using the CIFAR-10 test set.

Evaluation included:

* Test loss
* Test accuracy
* Confusion matrix
* Classification report
* Class-wise performance analysis
* Sample prediction visualization

## ⚠️ Important Limitation

The model is trained on CIFAR-10 images with a resolution of `32 × 32` pixels.

Because of this, predictions on ordinary internet photographs may be less reliable than predictions on images that closely resemble the CIFAR-10 dataset. A model can sometimes make an incorrect prediction even when its confidence score is high.

The reported **68.82% test accuracy** is measured on the CIFAR-10 test set and should not be interpreted as accuracy on arbitrary real-world photographs.

## 🔮 Future Improvements

A future Version 2 can focus on improving model performance through:

* Improved CNN architecture
* More effective data augmentation
* Hyperparameter tuning
* Learning-rate optimization
* Regularization improvements
* Improved generalization to varied images
* Additional model evaluation and calibration

## 👨‍💻 Author

**Abdullah Butt**

AI & Python Developer

GitHub:
https://github.com/abdullah8090butt-max

## 📜 License

This project is intended for educational, learning, and portfolio purposes.
