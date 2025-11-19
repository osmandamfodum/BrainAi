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
from scipy.stats import entropy  # For improved validation

# ========================= MODERN CSS DESIGN (GROK-LIKE) =========================
st.markdown("""
<style>
    /* Overall modern dark/light theme */
    .stApp { background-color: #f0f2f6; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #ddd; }
    .stButton > button { background-color: #4CAF50; color: white; border-radius: 8px; }
    .stDownloadButton > button { background-color: #2196F3; color: white; border-radius: 8px; }
    .stMetric { background-color: white; border: 1px solid #ddd; border-radius: 8px; padding: 10px; }
    .stAlert { border-radius: 8px; }
    .stChatMessage { border-radius: 12px; padding: 10px; margin-bottom: 10px; }
    .user { background-color: #e0f7fa; }
    .assistant { background-color: #fff; border: 1px solid #ddd; }
    /* Smooth transitions */
    * { transition: all 0.3s ease; }
    /* Dark mode toggle */
    .dark-mode .stApp { background-color: #1e1e1e; color: #fff; }
    .dark-mode [data-testid="stSidebar"] { background-color: #2a2a2a; border-right: 1px solid #444; }
    .dark-mode .stButton > button { background-color: #66BB6A; }
    .dark-mode .stDownloadButton > button { background-color: #42A5F5; }
    .dark-mode .stMetric { background-color: #2a2a2a; border: 1px solid #444; }
    .dark-mode .user { background-color: #006064; color: #fff; }
    .dark-mode .assistant { background-color: #333; border: 1px solid #555; color: #fff; }
</style>
""", unsafe_allow_html=True)
# Dark mode toggle (Grok-like)
dark_mode = st.sidebar.toggle("Dark Mode")
if dark_mode:
    st.markdown('<style>.dark-mode { display: block; }</style>', unsafe_allow_html=True)
# ========================= API KEY FROM ENV (FIXED) =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found! Ensure it's set in Streamlit Secrets.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')
# ========================= LOAD MODEL =========================
@st.cache_resource
def load_model():
    # This finds the model in the same folder as app.py (src/)
    model_path = os.path.join(os.path.dirname(__file__), "Model2.h5")  # ← change name if needed
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found!\n"
                 f"Expected: {model_path}\n"
                 f"Current folder contains: {os.listdir(os.path.dirname(__file__))}")
        st.stop()
    
    st.success(f"Model loaded: {os.path.basename(model_path)}")
    return tf.keras.models.load_model(model_path, compile=False)
model = load_model()
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
# ========================= IMAGE VALIDATOR (IMPROVED) =========================
def is_brain_mri(img):
    try:
        arr = np.array(img.resize((224, 224)))
        if arr.shape[-1] == 3:
            r, g, b = arr.mean(axis=(0,1))
            if not (abs(r-g) < 30 and abs(g-b) < 30):
                return False, "Color image detected. Brain MRIs are grayscale."
        
        # Intensity range check: MRIs have moderate variance (not too uniform or noisy)
        gray = np.array(img.convert('L'))
        if np.std(gray) < 10 or np.std(gray) > 100:  # Tune these thresholds based on your dataset
            return False, "Unusual intensity variance. Not typical for MRI."
        
        # Entropy check: MRIs have medium entropy (structured detail, not random)
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0,255))
        ent = entropy(hist + 1e-10)  # Avoid log(0)
        if ent < 3.0 or ent > 6.5:  # Low: too uniform (e.g., blank); High: too noisy (e.g., natural photo)
            return False, "Image entropy not in MRI range."
        
        # MobileNetV2 classification with expanded blocked list
        mob = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(arr.copy().astype('float32'))
        x = np.expand_dims(x, 0)
        preds = mob.predict(x, verbose=0)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0]
        label, conf = decoded[0][1].lower(), decoded[0][2]
        
        blocked = ['cat', 'dog', 'car', 'person', 'pizza', 'bird', 'flower', 'house', 'tree', 'food', 'phone', 'landscape', 'building', 'sky', 'mountain', 'ocean', 'fruit', 'vegetable', 'book', 'computer', 'screen', 'art', 'painting', 'drawing']
        if any(w in label for w in blocked) and conf > 0.3:
            return False, f"This appears to be a '{label.replace('_', ' ')}' (conf: {conf:.1%}), not an MRI scan."
        
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
   
    overall = max(results, key=lambda x: x[1])[0]
    pdf.cell(0, 10, f"• Most likely diagnosis: {overall}", ln=1)
    pdf.cell(0, 10, f"• Total slices analyzed: {len(images)}", ln=1)
    pdf.ln(5)
   
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Slice-by-Slice Results:", ln=1)
    pdf.set_font("Arial", size=10)
   
    for i, (img, (label, conf, _)) in enumerate(zip(images, results), 1):
        pdf.cell(0, 8, f" Slice {i}: {label} ({conf:.1%} confidence)", ln=1)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp.name)
            pdf.image(tmp.name, w=180)
            os.unlink(tmp.name)
        pdf.ln(5)
   
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, "AI-generated. Consult a neurosurgeon for final diagnosis.", ln=1)
   
    return pdf
# ========================= GROK-LIKE UI WITH HISTORY =========================
st.title("🧠 Brain MRI Tumor AI Analyzer")
st.markdown("**Upload MRI slices • Get AI diagnosis • Chat with Neurologist AI**")
# Sidebar for history (Grok-like recent activity)
st.sidebar.title("Recent Upload History")
if "upload_history" not in st.session_state:
    st.session_state.upload_history = []
for hist in st.session_state.upload_history[-5:]: # Show last 5
    st.sidebar.markdown(f"• {hist['date']} - {hist['diagnosis']} ({len(hist['slices'])} slices)")
st.sidebar.markdown("---")
st.sidebar.caption("History is per session (refreshes on reload).")
# Patient info
with st.expander("Patient Info for Report"):
    patient_name = st.text_input("Patient Name", "Anonymous")
    doctor_name = st.text_input("Doctor Name", "Grok AI Assistant")
# Upload
uploaded_files = st.file_uploader("Upload Brain MRI Slices (multiple OK)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if uploaded_files:
    images = []
    results = []
    valid_count = 0
    for file in uploaded_files:
        img = Image.open(file).convert("L").convert("RGB")
        with st.spinner(f"Validating {file.name}..."):
            valid, msg = is_brain_mri(img)
            if not valid:
                st.error(f"Rejected: {file.name} - {msg}")
                continue
       
        st.image(img, caption=f"Valid: {file.name}", width=150)
        label, conf, probs = predict_single(img)
        results.append((label, conf, probs))
        images.append(img)
        valid_count += 1
    if valid_count == 0:
        st.error("No valid MRIs uploaded.")
        st.stop()
    final_diagnosis = max(results, key=lambda x: x[1])[0]
    avg_conf = np.mean([r[1] for r in results])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Valid Slices", valid_count)
    with col2:
        st.metric("Diagnosis", final_diagnosis)
    with col3:
        st.metric("Avg Conf", f"{avg_conf:.1%}")
    if final_diagnosis != "No Tumor":
        st.error("Tumor Detected – See neurosurgeon urgently")
    else:
        st.success("No tumor detected")
    # Update history
    st.session_state.upload_history.append({
        "date": datetime.now().strftime("%H:%M"),
        "diagnosis": final_diagnosis,
        "slices": [f.name for f in uploaded_files]
    })
    # PDF
    with st.spinner("Generating PDF..."):
        pdf = create_pdf_report(images, results, patient_name, doctor_name)
        pdf_buffer = pdf.output(dest='S').encode('latin1')
    st.download_button(
        "📄 Download PDF Report",
        pdf_buffer,
        file_name=f"MRI_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf"
    )
# Chatbot (Grok-like full chat UI)
st.markdown("---")
st.subheader("Chat with AI Neurologist")
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Upload MRIs above or ask me anything about brain tumors."}]
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍⚕️" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
if prompt := st.chat_input("Ask about your MRI or brain health..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
   
    with st.chat_message("assistant", avatar="🧑‍⚕️"):
        with st.spinner("Thinking..."):
            reply = gemini_model.generate_content(
                f"Act as a neurosurgeon. User asks: {prompt}. Keep it professional, bullet points if needed. End with consult doctor."
            ).text
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
st.caption("AI tool for education • Consult a doctor for real advice")
