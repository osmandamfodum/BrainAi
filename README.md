# Brain Tumor MRI Analysis with AI Assistant

## Description
This repository contains a Streamlit-based web application that leverages a Deep Learning model (Keras/TensorFlow) to classify brain MRI scans into four categories. It integrates the Google Gemini Pro API to provide a conversational AI assistant that helps interpret results and suggests clinical next steps.

## Dataset Information
The model is trained to recognize the following classes:
* Glioma Tumor
* Meningioma Tumor
* Pituitary Tumor
* No Tumor (Healthy)
*Note: Ensure your dataset consists of T1 or T2 weighted axial/sagittal slices for optimal accuracy.*

## Requirements
* Python 3.9+
* TensorFlow 2.x
* Streamlit
* Pillow (PIL)
* NumPy
* google-generativeai

## Usage Instructions
1. **Clone the repository:** `git clone <your-repo-link>`
2. **Install Dependencies:**
   `pip install -r requirements.txt`
3. **Setup Secrets:**
   Create a folder `.streamlit` and a file `secrets.toml`. Add your API key:
   `GEMINI_API_KEY = "YOUR_KEY_HERE"`
4. **Run the App:**
   `streamlit run app.py`

## Methodology
The application uses a pre-trained CNN (`Model2.h5`). It implements a custom HSV-based saturation check to ensure uploaded images are grayscale (MRI) before processing. Images are resized to 299x299 and normalized before inference.
