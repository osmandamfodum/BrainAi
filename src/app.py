# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
from fpdf import FPDF
from datetime import datetime
import os
import tempfile
from scipy.stats import entropy

# ========================= CSS (GROK STYLE) =========================
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #ddd; }
    .stButton > button { background-color: #4CAF50; color: white; border-radius: 8px; }
    .stDownloadButton > button { background-color: #2196F3; color: white; border-radius: 8px; }
    .stMetric { background-color: white; border: 1px solid #ddd; border-radius: 8px; padding: 10px; }
    .stAlert { border-radius: 8px; }
    .stChatMessage { border-radius: 12px; padding: 10px; margin-bottom: 10px; }
    .user { background-color: #e0f7fa; }
    .assistant { background-color: #fff; border: 1px solid #ddd; }
    * { transition: all 0.3s ease; }
</style>
""", unsafe_allow_html=True)

dark_mode = st.sidebar.toggle("Dark Mode")
if dark_mode:
    st.markdown("<style>.stApp{background:#1e1e1e;color:white;}[data-testid='stSidebar']{background:#2a2a2a;}</style>", unsafe_allow_html=True)

# ========================= GEMINI API =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API key missing! Add it in Streamlit Secrets.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# ========================= LOAD MODEL (FIXED) =========================
@st.cache_resource
def load_model():
    # Your model filename is Model2.h5 (capital M!)
    model_path = os.path.join(os.path.dirname(__file__), "Model2.h5")
    
    if not os.path.exists(model_path):
        st.error(f"Model2.h5 not found!\n"
                 f"Files in folder: {os.listdir(os.path.dirname(__file__))}")
        st.stop()
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error("Failed to load model. Likely TensorFlow version issue.")
        st.code(str(e))
        st.stop()

model = load_model()
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ========================= IMAGE VALIDATOR =========================
def is_brain_mri(img):
    try:
        arr = np.array(img.resize((224, 224)))
        if arr.shape[-1] == 3:
            r, g, b = arr.mean(axis=(0,1))
            if not (abs(r-g) < 30 and abs(g-b) < 30):
                return False, "Color image. MRIs are grayscale."
        gray = np.array(img.convert("L"))
        if np.std(gray) < 10 or np.std(gray) > 100:
            return False, "Unusual intensity."
        hist, _ = np.histogram(gray.flatten(), bins=256)
        ent = entropy(hist + 1e-10)
        if ent < 3 or ent > 6.5:
            return False, "Entropy not MRI-like."
        return True, "Valid MRI"
    except:
        return False, "Invalid image"

# ========================= PREDICTION =========================
def predict_single(img):
    img_resized = img.resize((299, 299))
    x = np.array(img_resized).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(pred))
    return class_names[idx], float(pred[idx]), pred.tolist()

# ========================= PDF REPORT =========================
def create_pdf_report(images, results, patient_name="Anonymous", doctor_name="AI Assistant"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Brain MRI Tumor Analysis Report", ln=1, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
    pdf.cell(0, 10, f"Patient: {patient_name}", ln=1)
    pdf.cell(0, 10, f"Referring: {doctor_name}", ln=1)
    pdf.ln(10)
    final = max(results, key=lambda x: x[1])[0]
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Most likely diagnosis: {final}", ln=1)
    pdf.cell(0, 10, f"Slices analyzed: {len(images)}", ln=1)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Slice Results:", ln=1)
    pdf.set_font("Arial", size=10)
    for i, (img, (label, conf, _)) in enumerate(zip(images, results), 1):
        pdf.cell(0, 8, f"Slice {i}: {label} ({conf:.1%})", ln=1)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            img.save(f.name)
            pdf.image(f.name, w=180)
            os.unlink(f.name)
        pdf.ln(5)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, "AI-generated report. Consult a neurosurgeon.", ln=1)
    return pdf

# ========================= MAIN UI =========================
st.title("Brain MRI Tumor AI Analyzer")
st.markdown("**Upload slices • Instant AI diagnosis • Professional report**")

with st.expander("Patient Info (for report)"):
    patient_name = st.text_input("Name", "Anonymous")
    doctor_name = st.text_input("Referring Doctor", "AI Assistant")

uploaded = st.file_uploader("Upload Brain MRI slices", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded:
    images = []
    results = []
    for file in uploaded:
        img = Image.open(file).convert("RGB")
        valid, msg = is_brain_mri(img)
        if not valid:
            st.error(f"{file.name}: {msg}")
            continue
        st.image(img, caption=file.name, width=150)
        label, conf, probs = predict_single(img)
        images.append(img)
        results.append((label, conf, probs))

    if images:
        final = max(results, key=lambda x: x[1])[0]
        avg_conf = np.mean([r[1] for r in results])
        c1, c2, c3 = st.columns(3)
        c1.metric("Valid Slices", len(images))
        c2.metric("Diagnosis", final)
        c3.metric("Confidence", f"{avg_conf:.1%}")

        if final != "No Tumor":
            st.error("Tumor detected — Urgent neurosurgery consult recommended")
        else:
            st.success("No tumor detected")

        pdf = create_pdf_report(images, results, patient_name, doctor_name)
        st.download_button("Download PDF Report", pdf.output(dest="S").encode("latin1"),
                           f"MRI_Report_{datetime.now().strftime('%Y%m%d')}.pdf", "application/pdf")

# ========================= CHAT =========================
st.markdown("---")
st.subheader("Ask AI Neurologist")
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload MRI or ask about brain tumors."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = gemini_model.generate_content(
                f"Neurosurgeon answer: {prompt}\nKeep short, professional. End with: Consult your doctor."
            ).text
        st.write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

st.caption("Educational AI tool • Always consult a medical professional")
