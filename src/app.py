# app.py
import streamlit as st
import numpy as np
from PIL import Image
import google.generativeai as genai
from fpdf import FPDF
from datetime import datetime
import os
import tempfile
from scipy.stats import entropy
import tflite_runtime.interpreter as tflite

# ========================= CSS =========================
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #ddd; }
    .stButton > button { background-color: #4CAF50; color: white; border-radius: 8px; }
    .stDownloadButton > button { background-color: #2196F3; color: white; border-radius: 8px; }
    .stChatMessage { border-radius: 12px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ========================= DARK MODE =========================
dark_mode = st.sidebar.toggle("Dark Mode")
if dark_mode:
    st.markdown("<style>.stApp{background:#1e1e1e;color:white;}</style>", unsafe_allow_html=True)

# ========================= API KEY =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found!")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ========================= LOAD TFLITE MODEL =========================
@st.cache_resource
def load_tflite():
    model_path = os.path.join(os.path.dirname(__file__), "model.tflite")

    if not os.path.exists(model_path):
        st.error(f"model.tflite not found in {os.path.dirname(__file__)}")
        st.stop()

    interpreter = tflite.Interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_tflite()

class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# ========================= VALIDATION =========================
def is_brain_mri(img):
    try:
        arr = np.array(img.resize((224, 224)))
        if arr.shape[-1] == 3:
            r, g, b = arr.mean(axis=(0,1))
            if not (abs(r-g) < 30 and abs(g-b) < 30):
                return False, "Color image detected. MRIs should be grayscale."
        gray = np.array(img.convert('L'))
        if np.std(gray) < 10 or np.std(gray) > 100:
            return False, "Unusual intensity variance."
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0,255))
        ent = entropy(hist + 1e-10)
        if ent < 3.0 or ent > 6.5:
            return False, "Image entropy not typical for MRI."
        return True, "Valid brain MRI"
    except:
        return False, "Invalid image format."

# ========================= PREDICT =========================
def predict_single(img):
    img_resized = img.resize((299, 299))
    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = np.argmax(pred)

    return class_names[idx], float(np.max(pred)), pred.tolist()

# ========================= PDF REPORT =========================
def create_pdf_report(images, results, patient_name="Patient", doctor_name="AI Assistant"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Brain MRI Tumor Analysis Report", ln=1, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
    pdf.cell(0, 10, f"Patient: {patient_name}", ln=1)
    pdf.cell(0, 10, f"Referring AI: {doctor_name}", ln=1)
    pdf.ln(10)

    overall = max(results, key=lambda x: x[1])[0]
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Most likely diagnosis: {overall}", ln=1)
    pdf.cell(0, 10, f"Total slices analyzed: {len(images)}", ln=1)

    for i, (img, (label, conf, _)) in enumerate(zip(images, results), start=1):
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 8, f"Slice {i}: {label} ({conf:.1%})", ln=1)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp.name)
            pdf.image(tmp.name, w=180)
            os.unlink(tmp.name)

    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, "AI-generated. Consult a neurosurgeon.", ln=1)
    return pdf

# ========================= UI START =========================
st.title("🧠 Brain MRI Tumor AI Analyzer")
st.markdown("Upload MRI slices and get AI-based analysis.")

uploaded_files = st.file_uploader("Upload MRI Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    images = []
    results = []

    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        ok, msg = is_brain_mri(img)

        if not ok:
            st.error(f"{file.name}: {msg}")
            continue

        label, conf, probs = predict_single(img)
        images.append(img)
        results.append((label, conf, probs))

        st.image(img, caption=f"{label} ({conf:.1%})", width=200)

    if not results:
        st.error("No valid MRI slices.")
        st.stop()

    diagnosis = max(results, key=lambda x: x[1])[0]
    st.success(f"Final Diagnosis: **{diagnosis}**")

    pdf = create_pdf_report(images, results)
    pdf_data = pdf.output(dest="S").encode("latin1")

    st.download_button("📄 Download PDF", pdf_data, "report.pdf", "application/pdf")

# ========================= CHAT =========================
st.subheader("Chat with AI Neurologist")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload MRI images or ask questions."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask something...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    reply = gemini_model.generate_content(prompt).text
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)
