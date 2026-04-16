# 🌾 Rice Leaf Disease Detection using CNN

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-GPU%20Enabled-yellow?style=for-the-badge&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A Deep Learning project for automatic detection of rice leaf diseases using Convolutional Neural Networks (CNN)**

*Digital Image Processing Course Project*

[🚀 Open in Colab](https://colab.research.google.com/drive/1zrNxfprjO6Cnw2ez3pn0M4a0s2F4EH82#scrollTo=uFRtBt4J8H72) · [📊 Dataset](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases) · [📁 Repository](https://github.com/MinarIm/Rice-Leaf-Disease-Detection-CNN-Project)

</div>

---

## 📌 Project Overview

Rice is one of the most important food crops in the world. Diseases in rice leaves can severely reduce crop yield and quality. Early and accurate detection of these diseases is critical for farmers to take timely action.

This project builds a **Convolutional Neural Network (CNN)** model from scratch using **TensorFlow/Keras** to automatically classify rice leaf images into three disease categories. The model learns visual patterns from leaf images and predicts the disease type with high confidence — enabling fast, automated diagnosis without the need for agricultural experts.

---

## 🎯 Objectives

- Build a CNN model capable of classifying rice leaf diseases from images
- Apply image preprocessing and data augmentation to handle small dataset size
- Evaluate model performance using accuracy, loss curves, and confusion matrix
- Deploy a reusable prediction function for real-world image inputs

---

## 🦠 Disease Classes

The model detects the following **3 rice leaf diseases**:

| # | Disease | Description |
|---|---------|-------------|
| 1 | 🟡 **Bacterial Leaf Blight** | Water-soaked to yellowish stripes on leaf margins, caused by *Xanthomonas oryzae* |
| 2 | 🟤 **Brown Spot** | Circular brown spots with yellow halos on leaves, caused by *Bipolaris oryzae* |
| 3 | ⚫ **Leaf Smut** | Small, slightly raised black spots on both leaf surfaces, caused by *Entyloma oryzae* |

---

## 📁 Dataset

| Property | Details |
|----------|---------|
| **Source** | [Kaggle - Rice Leaf Diseases by vbookshelf](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases) |
| **Total Images** | 120 images |
| **Images per Class** | 40 images each |
| **Image Format** | JPEG (.jpg / .JPG) |
| **Input Size** | Resized to 224 × 224 pixels |

```
rice_leaf_diseases/
├── Bacterial leaf blight/    → 40 images
├── Brown spot/               → 40 images
└── Leaf smut/                → 40 images
```

> **Note:** The dataset is very small (120 images total), so **Data Augmentation** was applied during training to artificially expand the training set and prevent overfitting.

---

## 🧠 Model Architecture

A custom CNN model was built from scratch with 4 convolutional blocks:

```
Input Image (224 × 224 × 3)
        │
┌───────▼────────┐
│  Conv2D (32)   │  ← Detects basic features: edges, colors
│  BatchNorm     │
│  MaxPooling    │
└───────┬────────┘
        │
┌───────▼────────┐
│  Conv2D (64)   │  ← Detects intermediate patterns
│  BatchNorm     │
│  MaxPooling    │
└───────┬────────┘
        │
┌───────▼────────┐
│  Conv2D (128)  │  ← Detects disease-specific textures
│  BatchNorm     │
│  MaxPooling    │
└───────┬────────┘
        │
┌───────▼────────┐
│  Conv2D (256)  │  ← High-level feature extraction
│  BatchNorm     │
│  MaxPooling    │
└───────┬────────┘
        │
   Flatten Layer
        │
  Dense (512) + Dropout (0.5)
        │
  Dense (256) + Dropout (0.3)
        │
  Dense (3) → Softmax
        │
   Output: [Bacterial Blight, Brown Spot, Leaf Smut]
```

### Key Design Choices

- **BatchNormalization** — Stabilizes and speeds up training
- **MaxPooling** — Reduces spatial dimensions while keeping important features
- **Dropout (0.5 & 0.3)** — Prevents overfitting on small dataset
- **Softmax output** — Gives probability for each of the 3 disease classes

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Input Image Size | 224 × 224 |
| Batch Size | 8 |
| Epochs | 30 (with Early Stopping) |
| Optimizer | Adam (lr = 0.001) |
| Loss Function | Categorical Crossentropy |
| Train/Val Split | 80% / 20% |

---

## 🔄 Data Augmentation

Since the dataset has only 120 images, augmentation was applied to training data to prevent overfitting:

```python
ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values to [0, 1]
    rotation_range=20,       # Random rotation up to 20°
    width_shift_range=0.2,   # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,         # Shear transformation
    zoom_range=0.2,          # Random zoom
    horizontal_flip=True,    # Mirror images
)
```

---

## 🛠️ Callbacks Used

| Callback | Purpose |
|----------|---------|
| `ModelCheckpoint` | Saves the best model automatically based on `val_accuracy` |
| `EarlyStopping` | Stops training if no improvement for 10 epochs |
| `ReduceLROnPlateau` | Reduces learning rate by 50% if validation loss plateaus |

---

## 🛠️ Technologies & Libraries

| Library | Purpose |
|---------|---------|
| `TensorFlow / Keras` | Building and training the CNN model |
| `NumPy` | Array operations and numerical processing |
| `Matplotlib` | Plotting training curves and images |
| `Seaborn` | Confusion matrix heatmap visualization |
| `Scikit-learn` | Classification report and evaluation metrics |
| `Google Colab` | Cloud-based GPU training environment |
| `Google Drive` | Dataset storage and model saving |

---

## 🚀 How to Run This Project

### Step 1 — Open in Google Colab
Click the badge below to open directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zrNxfprjO6Cnw2ez3pn0M4a0s2F4EH82#scrollTo=uFRtBt4J8H72)

### Step 2 — Enable GPU
```
Runtime → Change runtime type → Hardware accelerator → GPU → Save
```

### Step 3 — Download the Dataset
Download from [Kaggle](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases) and upload to Google Drive.

### Step 4 — Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 5 — Set Your Dataset Path
```python
DATASET_PATH = '/content/drive/MyDrive/rice_leaf_diseases/rice_leaf_diseases'
```

### Step 6 — Run All Cells
```
Runtime → Run all
```

---

## 📊 Model Evaluation

The model is evaluated using:

- ✅ **Training & Validation Accuracy/Loss Curves** — Visualizes learning progress over epochs
- ✅ **Confusion Matrix** — Shows which diseases the model confuses with each other
- ✅ **Classification Report** — Precision, Recall, and F1-Score per class

---

## 🔍 Predict on a New Image

```python
predict_disease(
    '/content/drive/MyDrive/rice_leaf_diseases/rice_leaf_diseases/Brown spot/DSC_0186.JPG',
    model,
    class_names
)
```

**Sample Output:**
```
🔍 Result: Brown spot
📊 Confidence: 91.3%

All probabilities:
  Bacterial leaf blight: 4.2%
  Brown spot: 91.3%
  Leaf smut: 4.5%
```

---

## 📂 Repository Structure

```
Rice-Leaf-Disease-Detection-CNN-Project/
│
├── 📓 rice_disease_cnn.ipynb     ← Main Colab notebook (all code)
├── 📊 best_rice_model.h5         ← Saved trained model weights
├── 🖼️ training_curves.png        ← Accuracy & loss graphs
├── 🖼️ confusion_matrix.png       ← Confusion matrix heatmap
├── 🖼️ sample_images.png          ← Sample dataset images
└── 📄 README.md                  ← This file
```

---

## 💡 Key Learnings

- How CNN layers extract features from images hierarchically
- Importance of **data augmentation** when working with small datasets
- How **BatchNormalization** and **Dropout** prevent overfitting
- Using **callbacks** to automate training optimization
- End-to-end deep learning pipeline: data → model → evaluation → prediction

---

## 🔮 Future Improvements

- [ ] Use **Transfer Learning** (MobileNetV2 / VGG16) for higher accuracy
- [ ] Expand dataset with more rice disease images
- [ ] Deploy model as a **web app** using Flask or Streamlit
- [ ] Add more disease categories
- [ ] Implement **Grad-CAM** to visualize which parts of the leaf the model focuses on

---

## 🙏 Acknowledgements

- Dataset by **[vbookshelf on Kaggle](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases)**
- Built and trained using **Google Colab** (free GPU)
- Inspired by real-world agricultural AI applications for food security

---

<div align="center">



</div>
