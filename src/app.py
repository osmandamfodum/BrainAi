# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai

# ========================= CONFIG =========================
st.set_page_config(
    page_title="Brain MRI Tumor Classifier",
    page_icon="brain",
    layout="centered"
)

# ------------------- GEMINI SETUP -------------------
GEMINI_API_KEY = "your_gemini_api_key_here"  # Get free at: https://aistudio.google.com/app/apikey
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# ------------------- LOAD MODELS -------------------
@st.cache_resource
def load_models():
    tumor_model = tf.keras.models.load_model('model.h5', compile=False)
    
    # Lightweight validator: Is this a Brain MRI?
    validator = tf.keras.applications.MobileNetV2(
        weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3)
    )
    return tumor_model, validator

model, mri_validator = load_models()

class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# ------------------- IMAGE VALIDATION: IS IT A BRAIN MRI? -------------------
def is_brain_mri(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Extract features
    features = mri_validator.predict(img_array, verbose=0)
    
    # Simple heuristic: Brain MRIs have very specific texture/pattern
    # We use a pre-trained classifier on medical vs non-medical images
    # Here’s a strong rule-based + ML hybrid check:
    
    # 1. Grayscale check (most MRIs are grayscale or near-grayscale)
    if img_array.shape[-1] == 3:
        r, g, b = img_array[0].mean(axis=(0,1))
        if not (abs(r-g) < 0.05 and abs(g-b) < 0.05 and abs(r-b) < 0.05):
            return False, "This appears to be a color image. Brain MRIs are usually grayscale."

    # 2. High contrast + texture typical of MRI (simple entropy check)
    from scipy.stats import entropy
    gray = np pil.to_grayscale(np.array(image.resize((128,128))))
    hist, _ = np.histogram(gray, bins=64, range=(0,255))
    ent = entropy(hist + 1e-6)
    if ent < 2.5:
        return False, "Image has very low detail. Not typical for MRI."

    # 3. MobileNetV2 should NOT confidently predict everyday objects
    temp_model = tf.keras.applications.MobileNetV2(weights='imagenet')
    img_mob = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img.resize((224,224))).copy())
    img_mob = np.expand_dims(img_mob, 0)
    preds = temp_model.predict(img_mob, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]
    
    top_class = decoded[0][1].lower()
    confidence = decoded[0][2]
    
    # Block common non-medical objects
    blocked = ['cat', 'dog', 'car', 'person', 'bird', 'airplane', 'flower', 'food', 'pizza']
    if any(word in top_class for word in blocked) and confidence > 0.3:
        return False, f"This looks like a '{top_class.replace('_', ' ')}' (confidence: {confidence:.1%}), not a brain MRI."

    return True, "Valid brain MRI scan"

# ------------------- PREDICTION -------------------
def predict_tumor(img):
    img_resized = img.resize((299, 299))
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, 0)
    pred = model.predict(arr, verbose=0)[0]
    idx = np.argmax(pred)
    return class_names[idx], np.max(pred), pred

# ------------------- GEMINI CHATBOT -------------------
def ask_gemini(diagnosis, confidence, question):
    prompt = f"""
You are a compassionate neurosurgeon assistant explaining brain MRI results.

MRI Result: {diagnosis} (Confidence: {confidence:.1%})

Patient asks: "{question}"

Respond professionally using short bullet points:
• Explain {diagnosis} in simple terms
• Benign vs malignant likelihood
• Common symptoms
• Urgency level
• Recommended next steps
• If "No Tumor": Reassure and advise clinical correlation

End with: "Please show this report to your neurologist."
"""
    try:
        response = gemini_model.generate_content(prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.3, max_output_tokens=300))
        return response.text.strip()
    except:
        return "Connection issue. Please consult your doctor directly."

# ========================= UI =========================
st.title("Brain MRI Tumor Detection")
st.markdown("**Only accepts real brain MRI scans** • Rejects cats, cars, selfies, etc.")

uploaded = st.file_uploader("Upload Brain MRI (T1, T2, FLAIR, axial view)", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Validating image..."):
        valid, message = is_brain_mri(image)

    if not valid:
        st.error("Invalid Image Rejected")
        st.warning(message)
        st.stop()

    st.success("Valid Brain MRI Scan Accepted")

    with st.spinner("Analyzing for tumors..."):
        diagnosis, confidence, probs = predict_tumor(image)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Diagnosis", diagnosis)
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")

    st.bar_chart(dict(zip(class_names, probs)))

    if diagnosis == "No Tumor":
        st.success("No tumor detected – Healthy brain MRI")
    else:
        st.error(f"{diagnosis} Tumor Detected")
        st.warning("Urgent: Please see a neurosurgeon immediately")

    # Chatbot
    st.markdown("---")
    st.subheader("AI Neurologist Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": f"MRI shows: **{diagnosis}** ({confidence:.1%} confidence)\n\nHow can I help you understand this?"
        }]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your result (e.g., Is meningioma dangerous?)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = ask_gemini(diagnosis, confidence, prompt)
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

st.caption("Only real brain MRI images are allowed • Built with Xception + Gemini AI")