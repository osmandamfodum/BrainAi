# app.py (ONNX Version)
import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
import google.generativeai as genai
from fpdf import FPDF
from datetime import datetime
import os
import tempfile
from scipy.stats import entropy  # For improved validation

# ========================= MODERN CSS DESIGN (GROK-LIKE) =========================
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

# ========================= DARK MODE =========================
dark_mode = st.sidebar.toggle("Dark Mode")
if dark_mode:
    st.markdown("""
        <style>
            .stApp { background-color: #1e1e1e; color: white; }
            [data-testid="stSidebar"] { background-color: #2a2a2a; }
        </style>
    """, unsafe_allow_html=True)

# ========================= API KEY =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found in environment variables.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# ========================= LOAD ONNX MODEL =========================
@st.cache_resource
def load_onnx_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.onnx")

    if not os.path.exists(model_path):
        st.error(f"model.onnx not found in folder: {os.path.dirname(__file__)}")
        st.stop()

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session

session = load_onnx_model()
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ========================= IMAGE VALIDATOR =========================
def is_brain_mri(img):
    try:
        arr = np.array(img.resize((224, 224)))

        # Grayscale validation
        if arr.shape[-1] == 3:
            r, g, b = arr.mean(axis=(0,1))
            if not (abs(r-g) < 30 and abs(g-b) < 30):
                return False, "Color image detected. MRIs are grayscale."

        # Intensity validation
        gray = np.array(img.convert("L"))
        if np.std(gray) < 10 or np.std(gray) > 100:
            return False, "Unusual intensity variance."

        # Entropy validation
        hist, _ = np.histogram(gray.flatten(), bins=256)
        ent = entropy(hist + 1e-10)
        if ent < 3 or ent > 6.5:
            return False, "Image entropy does not match MRI characteristics."

        return True, "Valid MRI"
    except:
        return False, "Invalid image"

# ========================= ONNX PREDICTION =========================
def predict_single(img):
    img_resized = img.resize((299, 299))
    x = np.array(img_resized).astype("float32") / 255.0
    x = np.expand_dims(x, 0)

    inputs = {session.get_inputs()[0].name: x}
    outputs = session.run(None, inputs)[0][0]

    idx = int(np.argmax(outputs))
    return class_names[idx], float(outputs[idx]), outputs.tolist()

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

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Summary of Findings:", ln=1)

    final = max(results, key=lambda x: x[1])[0]
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 10, f"Most likely diagnosis: {final}", ln=1)
    pdf.cell(0, 10, f"Total slices analyzed: {len(images)}", ln=1)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Slice-by-Slice:", ln=1)
    pdf.set_font("Arial", size=10)

    for i, (img, (label, conf, _)) in enumerate(zip(images, results), start=1):
        pdf.cell(0, 8, f"Slice {i}: {label} ({conf:.1%})", ln=1)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp.name)
            pdf.image(tmp.name, w=180)
            os.unlink(tmp.name)

        pdf.ln(5)

    return pdf

# ========================= MAIN UI =========================
st.title("🧠 Brain MRI Tumor AI Analyzer")
st.markdown("**Upload MRI slices • Get AI diagnosis • Chat with Neurologist AI**")

# Upload
uploaded_files = st.file_uploader("Upload Brain MRI Slices", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    images = []
    results = []

    for file in uploaded_files:
        img = Image.open(file).convert("RGB")

        valid, msg = is_brain_mri(img)
        if not valid:
            st.error(f"{file.name}: {msg}")
            continue

        st.image(img, caption=file.name, width=150)
        label, conf, probs = predict_single(img)

        images.append(img)
        results.append((label, conf, probs))

    if not images:
        st.error("No valid MRIs detected.")
        st.stop()

    # Metrics
    final_diagnosis = max(results, key=lambda x: x[1])[0]
    avg_conf = np.mean([r[1] for r in results])

    c1, c2, c3 = st.columns(3)
    c1.metric("Valid Slices", len(images))
    c2.metric("Diagnosis", final_diagnosis)
    c3.metric("Avg Conf", f"{avg_conf:.1%}")

    # PDF
    pdf = create_pdf_report(images, results)
    pdf_bytes = pdf.output(dest="S").encode("latin1")

    st.download_button("📄 Download PDF Report", pdf_bytes, "MRI_Report.pdf", mime="application/pdf")

# ========================= CHATBOT =========================
st.markdown("---")
st.subheader("Chat with AI Neurologist")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload MRI images or ask anything about brain tumors."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about MRI or brain health..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        reply = gemini_model.generate_content(
            f"Act as a neurosurgeon. User asks: {prompt}. Reply professionally with short, clear bullet points. "
            "Always recommend consulting a real doctor."
        ).text
        st.write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

st.caption("AI tool for educational purposes • Always consult a medical professional.")
