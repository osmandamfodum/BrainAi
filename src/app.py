# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
from fpdf import FPDF
import base64
from datetime import datetime
import os
import tempfile

# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title="Brain MRI Tumor AI",
    page_icon="brain",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ========================= SECRETS (HUGGINGFACE) =========================
# On Hugging Face Spaces → Settings → Secrets → Add: GEMINI_API_KEY = your_key_here
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found! Add it in Secrets (Hugging Face) or environment.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# ========================= LOAD MODEL =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5', compile=False)

model = load_model()
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# ========================= IMAGE VALIDATOR =========================
def is_brain_mri(img):
    try:
        arr = np.array(img.resize((224, 224)))
        if arr.shape[-1] == 3:
            r, g, b = arr.mean(axis=(0,1))
            if not (abs(r-g) < 30 and abs(g-b) < 30):  # Not near-grayscale
                return False, "Color image detected. Brain MRIs are grayscale."
        
        # MobileNetV2 check for obvious non-medical content
        mob = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(arr.copy().astype('float32'))
        x = np.expand_dims(x, 0)
        preds = mob.predict(x, verbose=0)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0]
        label, conf = decoded[0][1], decoded[0][2]
        
        blocked = ['cat', 'dog', 'car', 'person', 'pizza', 'bird', 'flower']
        if any(w in label.lower() for w in blocked) and conf > 0.3:
            return False, f"This appears to be a '{label}' not an MRI scan."
        
        return True, "Valid brain MRI"
    except:
        return False, "Invalid image format."

# ========================= PREDICTION =========================
def predict_single(img):
    img_resized = img.resize((299, 299))
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, 0)
    pred = model.predict(arr, verbose=0)[0]
    idx = np.argmax(pred)
    return class_names[idx], float(np.max(pred)), pred.tolist()

# ========================= PDF REPORT =========================
def create_pdf_report(images, results, patient_name="Patient", doctor_name="AI Assistant"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Brain MRI Tumor Analysis Report", ln=1, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
    pdf.cell(0, 10, f"Patient: {patient_name}", ln=1)
    pdf.cell(0, 10, f"Referring AI: {doctor_name}", ln=1)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Summary of Findings:", ln=1)
    pdf.set_font("Arial", size=11)
    
    overall = max(results, key=lambda x: x[1])[0]  # Most confident prediction
    pdf.cell(0, 10, f"• Most likely diagnosis: {overall}", ln=1)
    pdf.cell(0, 10, f"• Total slices analyzed: {len(images)}", ln=1)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Slice-by-Slice Results:", ln=1)
    pdf.set_font("Arial", size=10)
    
    for i, (img, (label, conf, _)) in enumerate(zip(images, results), 1):
        pdf.cell(0, 8, f"  Slice {i}: {label} ({conf:.1%} confidence)", ln=1)
        # Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp.name)
            pdf.image(tmp.name, w=180)
            os.unlink(tmp.name)
        pdf.ln(5)
    
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, "This is an AI-generated report. Final diagnosis must be confirmed by a qualified neurosurgeon.", ln=1)
    
    return pdf

# ========================= MAIN UI =========================
st.title("Brain MRI Tumor Detection")
st.markdown("**Multi-slice • PDF Report • AI Neurologist Assistant**")

# Patient info (optional)
with st.expander("Patient Information (for report)", expanded=False):
    patient_name = st.text_input("Patient Name", "Anonymous Patient")
    doctor_name = st.text_input("Referring Doctor", "AI Assistant")

# Upload multiple images
uploaded_files = st.file_uploader(
    "Upload Brain MRI slices (multiple allowed)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    images = []
    results = []
    valid_count = 0

    for file in uploaded_files:
        img = Image.open(file).convert("L").convert("RGB")  # Force grayscale → RGB
        with st.spinner(f"Validating {file.name}..."):
            valid, msg = is_brain_mri(img)
            if not valid:
                st.error(f"Rejected {file.name}")
                st.warning(msg)
                continue
        
        st.image(img, caption=f"Valid MRI: {file.name}", width=200)
        label, conf, probs = predict_single(img)
        results.append((label, conf, probs))
        images.append(img)
        valid_count += 1

    if valid_count == 0:
        st.error("No valid MRI images uploaded.")
        st.stop()

    # Overall result
    final_diagnosis = max(results, key=lambda x: x[1])[0]
    avg_conf = np.mean([r[1] for r in results])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Valid Slices", valid_count)
    with col2:
        st.metric("Final Diagnosis", final_diagnosis)
    with col3:
        st.metric("Avg Confidence", f"{avg_conf:.1%}")

    if final_diagnosis != "No Tumor":
        st.error("Tumor Detected – Urgent neurosurgery consultation recommended")
    else:
        st.success("No tumor detected across all slices")

    # PDF Download
    with st.spinner("Generating medical report..."):
        pdf = create_pdf_report(images, results, patient_name, doctor_name)
        pdf_buffer = pdf.output(dest='S').encode('latin1')
        b64 = base64.b64encode(pdf_buffer).decode()

    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name=f"MRI_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf"
    )

    # Chatbot
    st.markdown("---")
    st.subheader("AI Neurologist Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": f"Analysis complete: **{final_diagnosis}** in {valid_count} slice(s).\n\nHow can I help you understand this result?"
        }]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your MRI result..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Consulting AI neurologist..."):
                reply = gemini_model.generate_content(
                    f"Brain MRI shows {final_diagnosis} across {valid_count} slices. Patient asks: {prompt}\n"
                    "Respond professionally in bullet points. End with: Please consult your neurosurgeon."
                ).text
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

st.caption("Professional AI tool • Only accepts real brain MRI • PDF report ready")