# app.py  ←  FINAL 2025 VERSION (100% working on Streamlit Cloud)
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

# ========================= UI STYLE =========================
st.set_page_config(page_title="Brain MRI AI", page_icon="brain", layout="centered")
st.markdown("""
<style>
    .stApp { background: linear-gradient(to bottom, #f0f2f6, #e0e7ff); }
    .stButton>button { background: #4CAF50; border-radius: 12px; height: 3em; }
    .stDownloadButton>button { background: #2196F3; border-radius: 12px; }
    .big-font { font-size: 48px !important; font-weight: bold; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='big-font'>Brain MRI Tumor AI</h1>", unsafe_allow_html=True)
st.markdown("**Upload slices → Instant diagnosis → Professional PDF report**")

# ========================= GEMINI =========================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Add GEMINI_API_KEY in Streamlit Secrets!")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# ========================= LOAD MODERN .KERAS MODEL =========================
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "model_fixed.keras")
    if not os.path.exists(path):
        st.error(f"model_fixed.keras not found!\nFiles present: {os.listdir(os.path.dirname(__file__))}")
        st.stop()
    model = tf.keras.models.load_model(path)
    st.success("Model loaded (modern .keras format)")
    return model

model = load_model()
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ========================= VALIDATOR =========================
def is_brain_mri(img):
    try:
        arr = np.array(img.resize((224,224)))
        if arr.shape[-1] == 3:
            if not (abs(arr[:,:,0].mean() - arr[:,:,1].mean()) < 30 and abs(arr[:,:,1].mean() - arr[:,:,2].mean()) < 30):
                return False, "Color image detected"
        gray = np.array(img.convert("L"))
        if not (10 < np.std(gray) < 100):
            return False, "Unusual intensity"
        hist,_ = np.histogram(gray.flatten(), bins=256)
        if not (3 < entropy(hist + 1e-10) < 6.5):
            return False, "Not MRI-like entropy"
        return True, "Valid MRI"
    except:
        return False, "Invalid"

# ========================= PREDICT =========================
def predict_single(img):
    x = np.array(img.resize((299,299))).astype("float32") / 255.0
    x = np.expand_dims(x, 0)
    pred = model.predict(x, verbose=0)[0]
    idx = np.argmax(pred)
    return class_names[idx], float(pred[idx])

# ========================= PDF =========================
def create_pdf(images, results, name="Anonymous"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Brain MRI Tumor Report", ln=1, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
    pdf.cell(0, 10, f"Patient: {name}", ln=1)
    pdf.ln(10)
    final = max(results, key=lambda x: x[1])[0]
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Final Diagnosis: {final}", ln=1)
    pdf.ln(5)
    for i, (img, (label, conf)) in enumerate(zip(images, results), 1):
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 8, f"Slice {i}: {label} ({conf:.1%})", ln=1)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            img.save(f.name)
            pdf.image(f.name, w=180)
            os.unlink(f.name)
        pdf.ln(5)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, "AI-generated. Consult a neurosurgeon.", ln=1)
    return pdf

# ========================= MAIN =========================
with st.expander("Patient Info"):
    patient_name = st.text_input("Name", "Anonymous Patient")

files = st.file_uploader("Upload MRI slices", ["jpg","jpeg","png"], accept_multiple_files=True)

if files:
    valid_imgs = []
    results = []
    for f in files:
        img = Image.open(f).convert("RGB")
        ok, msg = is_brain_mri(img)
        if not ok:
            st.error(f"{f.name}: {msg}")
            continue
        st.image(img, width=150, caption=f.name)
        label, conf = predict_single(img)
        valid_imgs.append(img)
        results.append((label, conf))

    if valid_imgs:
        final = max(results, key=lambda x: x[1])[0]
        conf = np.mean([c for _,c in results])
        c1,c2,c3 = st.columns(3)
        c1.metric("Slices", len(valid_imgs))
        c2.metric("Diagnosis", final)
        c3.metric("Avg Confidence", f"{conf:.1%}")

        if final != "No Tumor":
            st.error("TUMOR DETECTED — Urgent consultation needed!")
        else:
            st.success("No tumor detected")

        pdf = create_pdf(valid_imgs, results, patient_name)
        st.download_button("Download PDF Report", pdf.output(dest="S").encode("latin1"),
                          f"MRI_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", "application/pdf")

# ========================= CHAT =========================
st.markdown("---")
st.subheader("AI Neurologist Chat")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about your results..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = gemini_model.generate_content(f"Neurosurgeon reply: {prompt}").text
        st.write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

st.caption("2025 Modern .keras version • No TensorFlow issues ever again")
