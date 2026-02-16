# Brain Tumor MRI Analysis with AI Assistant

## Description
This repository contains a Streamlit-based web application that leverages a Deep Learning model (Keras/TensorFlow) to classify brain MRI scans into four categories. It integrates the Google Gemini Pro API to provide a conversational AI assistant that helps interpret results and suggests clinical next steps.

## Dataset Information
The model is trained to recognize the following classes:
* Glioma Tumor
* Meningioma Tumor
* Pituitary Tumor
* No Tumor (Healthy)
## How to load the dataset in colab 
* ## STEP1
* First loging into your kaggle account  
* under profile tab download the token file (kaggle.json)
* ## STEP2
* import it into colab
* then copy this code and paste it into colab cell '!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json' then run the cell
* ## STEP3
* Now run this code ' !kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p /content/dataset' it will download the dataset in colab this is faster way to import dataset to colab you can use it for any dataset as well just make sure you have kaggle.json file and you run the code in step 2
* ## STEP4
* Now after dataset downloaded then you need to unzip it use this code to do this step ' import zipfile
import os

zip_path = '/content/dataset/brain-tumor-mri-dataset.zip'
extract_path = '/content/extracted'

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Unzipping completed!")
else:
    print(f"File not found: {zip_path}")'
* ## STEP5
* now you can import the brain.ipynb to colab and run the code just add the path of you dataset as it is in step 4
* ## STEP6
* after you run the code and saved the modle now you can fix the path in app.py to allow streamlit app read your model 
## Requirements
* Python 3.9+
* TensorFlow 2.x
* Streamlit
* Pillow (PIL)
* NumPy
* google-generativeai
* streamlit
* altair<5


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
The application uses a trained CNN (`Model2.h5`). It implements a custom HSV-based saturation check to ensure uploaded images are grayscale (MRI) before processing. Images are resized to 299x299 and normalized before inference.
