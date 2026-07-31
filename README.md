# Brain Tumor MRI Analysis with AI Assistant

## Description
This repository contains a Streamlit-based web application that leverages a Deep Learning model (Keras/TensorFlow) to classify brain MRI scans into four categories. It integrates the Google Gemini Pro API to provide a conversational AI assistant that helps interpret results and suggests clinical next steps.

## Dataset Information

The model was trained using the publicly available **Brain Tumor MRI Dataset** published on Kaggle.

**Dataset Details**

- **Dataset Name:** Brain Tumor MRI Dataset
- **Dataset Owner:** Masoud Nickparvar
- **Platform:** Kaggle
- **Dataset URL:** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- **License:** Refer to the dataset license on the Kaggle dataset page.

The dataset contains MRI brain images categorized into four classes:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor (Healthy)

**Data Availability Statement**

The MRI images used in this project were **not generated or collected by the authors**. They were obtained from the publicly available Brain Tumor MRI Dataset provided by **Masoud Nickparvar** on Kaggle.

The dataset is available at:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Any use of the dataset should comply with the terms and license specified by the dataset owner on Kaggle.

## How to load the dataset in Colab

### STEP 1
First log into your Kaggle account.

Under your profile, download your API token (`kaggle.json`).

### STEP 2
Upload `kaggle.json` to Google Colab and execute:

```python
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### STEP 3
Download the dataset:

```python
!kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p /content/dataset
```

### STEP 4
Extract the dataset:

```python
import zipfile
import os

zip_path = '/content/dataset/brain-tumor-mri-dataset.zip'
extract_path = '/content/extracted'

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Unzipping completed!")
else:
    print(f"File not found: {zip_path}")
```

### STEP 5
Import `brain.ipynb` into Google Colab and update the dataset path to the extracted directory.

### STEP 6
After training and saving the model, update the model path in `app.py` so the Streamlit application can load the trained model.

## Requirements

- Python 3.9+
- TensorFlow 2.x
- Streamlit
- Pillow (PIL)
- NumPy
- google-generativeai
- altair < 5

## Usage Instructions

1. Clone the repository:

```bash
git clone <your-repo-link>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY="YOUR_API_KEY"
```

4. Run the application:

```bash
streamlit run app.py
```

## Methodology

The application uses a trained CNN (`Model2.h5`) for four-class brain MRI classification.

Prior to inference, each uploaded image undergoes an HSV-based saturation analysis to verify that it is a grayscale MRI image. Images are resized to **299 × 299 pixels**, normalized, and passed through the trained TensorFlow/Keras model for prediction.

The predicted class is then provided to the integrated Google Gemini AI assistant, which generates an educational explanation of the prediction and suggests possible clinical next steps. The AI-generated responses are intended solely for research and educational purposes and should not be considered a medical diagnosis.
