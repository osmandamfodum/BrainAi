# Brain Tumor MRI Analysis with AI Assistant

## Description

This repository contains a Streamlit-based web application that leverages a Deep Learning model (TensorFlow/Keras) to classify brain MRI scans into four categories. The application integrates the Google Gemini API to provide an AI assistant that explains prediction results and offers educational information regarding possible clinical interpretations and recommended next steps.

---

## Code Availability

The complete source code, trained deep learning model, training notebook, and documentation associated with this project are permanently archived on **Zenodo** to ensure reproducibility and long-term accessibility.

**Zenodo DOI:**  
https://doi.org/10.5281/zenodo.21720678

The Zenodo archive includes:

- Streamlit application (`app.py`)
- Training notebook (`brain.ipynb`)
- Trained CNN model (`Model2.h5`)
- Project documentation (`README.md`)
- Dependency list (`requirements.txt`)

---

## Dataset Information

This project was developed using the publicly available **Brain Tumor MRI Dataset**.

### Dataset Details

- **Dataset Name:** Brain Tumor MRI Dataset
- **Dataset Owner:** Masoud Nickparvar
- **Platform:** Kaggle
- **Dataset URL:** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- **License:** Please refer to the dataset license available on the Kaggle dataset page.

The dataset contains MRI brain images belonging to the following four classes:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor (Healthy)

### Data Availability Statement

The MRI images used in this project were **not generated or collected by the authors**. They were obtained from the publicly available **Brain Tumor MRI Dataset** provided by **Masoud Nickparvar** through Kaggle.

The dataset is available at:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Users should download the dataset directly from Kaggle and comply with the dataset owner's license and terms of use.

---

## Loading the Dataset in Google Colab

### Step 1

Log in to your Kaggle account and download your API token (`kaggle.json`) from your account settings.

### Step 2

Upload `kaggle.json` to Google Colab and run:

```python
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### Step 3

Download the dataset:

```python
!kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p /content/dataset
```

### Step 4

Extract the downloaded dataset:

```python
import zipfile
import os

zip_path = "/content/dataset/brain-tumor-mri-dataset.zip"
extract_path = "/content/extracted"

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    print("Unzipping completed!")
else:
    print(f"File not found: {zip_path}")
```

### Step 5

Open `brain.ipynb` in Google Colab and update the dataset path to the extracted directory.

### Step 6

Train the model (or load the provided model), save the trained weights, and update the model path in `app.py` if necessary.

---

## Requirements

- Python 3.9+
- TensorFlow 2.x
- Streamlit
- Pillow (PIL)
- NumPy
- google-generativeai
- altair < 5

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Usage

### Clone the repository

```bash
git clone https://github.com/osmandamfodum/BrainAi.git
cd BrainAi
```

### Configure the Gemini API key

Create a directory named `.streamlit` and create a file named `secrets.toml`:

```toml
GEMINI_API_KEY="YOUR_API_KEY"
```

### Run the application

```bash
streamlit run app.py
```

---

## Methodology

The application employs a TensorFlow/Keras Convolutional Neural Network (`Model2.h5`) trained to classify brain MRI images into four diagnostic categories.

Before inference, each uploaded image undergoes a custom HSV-based saturation analysis to verify that it is a grayscale MRI image. Valid images are resized to **299 × 299 pixels**, normalized, and passed to the trained CNN for prediction.

The predicted class is subsequently sent to the integrated Google Gemini AI assistant, which provides an educational explanation of the prediction together with possible clinical considerations. The AI-generated responses are intended solely for **research and educational purposes** and **must not be interpreted as a medical diagnosis or a substitute for professional clinical judgment**.

---

## Citation

If you use this repository in your research, please cite the archived version:

> Osman Al-Hussein. *Brain Tumor MRI Analysis with AI Assistant*. Zenodo. https://doi.org/10.5281/zenodo.21720678
