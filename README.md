# 🧠 AI Image Classifier

A CNN-based image classification application trained on the **CIFAR-10 dataset** and deployed as an interactive **Streamlit web application**.

## 🚀 Live Demo

👉 **[Open the AI Image Classifier](https://ai-image-classifier-4ipjdqlnh8e4ecufskxyv3.streamlit.app/)**

## 🔗 Project Links

* **GitHub Repository:** [AI Image Classifier](https://github.com/abdullah8090butt-max/AI-image-classifier)
* **Live Streamlit App:** [Open Application](https://ai-image-classifier-4ipjdqlnh8e4ecufskxyv3.streamlit.app/)

---

## 📌 Project Overview

The **AI Image Classifier** is an image classification system built using a **Convolutional Neural Network (CNN)** and trained on the **CIFAR-10 dataset**.

The application allows users to upload an image through a Streamlit interface and receive a predicted class along with its confidence and probability distribution across all 10 supported categories.

### The application provides:

* 🎯 Predicted image class
* 📊 Prediction confidence
* 📈 Probability for all supported classes
* 🖼️ Image preprocessing
* 🔧 Technical preprocessing information
* 🌐 Live web-based deployment

---

## 🎯 Supported Classes

The model classifies images into the following **10 CIFAR-10 categories**:

1. ✈️ Airplane
2. 🚗 Automobile
3. 🐦 Bird
4. 🐱 Cat
5. 🦌 Deer
6. 🐶 Dog
7. 🐸 Frog
8. 🐴 Horse
9. 🚢 Ship
10. 🚚 Truck

---

## 🧠 Model

The project uses a **Convolutional Neural Network (CNN)** trained specifically for CIFAR-10 image classification.

### Model Input

* Image size: `32 × 32 × 3`
* Color format: RGB
* Pixel values: normalized from `0–255` to `0–1`

### Dataset

CIFAR-10 contains **60,000 labeled color images** distributed across 10 classes.

The dataset was divided into:

| Dataset Split |     Images |
| ------------- | ---------: |
| Training      |     45,000 |
| Validation    |      5,000 |
| Testing       |     10,000 |
| **Total**     | **60,000** |

---

## 📊 Model Performance

The final **Version 1** model achieved the following results on the CIFAR-10 test set:

| Metric            |     Result |
| ----------------- | ---------: |
| **Test Accuracy** | **68.82%** |
| **Test Loss**     | **0.9148** |
| Test Images       | **10,000** |

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

---

## ✨ Features

* 🧠 CNN-based image classification
* 📤 JPG, JPEG, and PNG image upload
* 🖼️ Automatic RGB conversion
* 📐 Automatic resizing to `32 × 32`
* 🔢 Pixel normalization
* 🎯 Predicted class display
* 📊 Confidence score
* 📈 Class probability visualization
* 🔧 Technical preprocessing details
* ⚠️ Confidence-level feedback
* 🌐 Streamlit Community Cloud deployment

---

## 🛠️ Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **NumPy**
* **Pillow**
* **Scikit-learn**
* **Streamlit**
* **Git**
* **GitHub**

---

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

---

## ⚙️ Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/abdullah8090butt-max/AI-image-classifier.git
```

### 2. Open the project directory

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

> **Note:** Python 3.13 is used for the deployed Streamlit environment because it is compatible with the TensorFlow version used by this project.

---

## 🔄 How the Application Works

```text
User uploads an image
        ↓
Convert image to RGB
        ↓
Resize image to 32 × 32
        ↓
Normalize pixel values
        ↓
Pass image to CNN model
        ↓
Generate class probabilities
        ↓
Select highest-probability class
        ↓
Display prediction and confidence
        ↓
Display probabilities for all classes
```

---

## 🧪 Model Evaluation

The model was evaluated using the **10,000-image CIFAR-10 test set**.

The evaluation process included:

* Test loss
* Test accuracy
* Confusion matrix
* Classification report
* Class-wise performance analysis
* Sample prediction visualization

---

## ⚠️ Important Limitation

This model was trained on **CIFAR-10 images at 32 × 32 pixels**.

Because real-world photographs can differ significantly from the training data in image size, composition, background, lighting, and visual patterns, predictions on ordinary internet photographs may be less reliable.

A model may also produce a high confidence score for an incorrect prediction.

Therefore, the reported **68.82% test accuracy** represents performance on the **CIFAR-10 test dataset** and should not be interpreted as accuracy on arbitrary real-world photographs.

---

## 🔮 Future Improvements

A future **Version 2** can focus on improving accuracy and generalization through:

* Improved CNN architecture
* More effective data augmentation
* Hyperparameter tuning
* Learning-rate optimization
* Regularization improvements
* Better handling of visually similar classes
* Improved generalization to varied images
* Confidence calibration and additional evaluation

---

## 👨‍💻 Author

**Abdullah Butt**

**AI & Python Developer**

### GitHub

[github.com/abdullah8090butt-max](https://github.com/abdullah8090butt-max)

### Project Repository

[AI Image Classifier](https://github.com/abdullah8090butt-max/AI-image-classifier)

---

## 📜 License

This project is intended for **educational, learning, and portfolio purposes**.
