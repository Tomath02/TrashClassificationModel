import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd

# -------------------------------------------------------------
# Streamlit – Waste Classification (Organik | Anorganik | B3)
# -------------------------------------------------------------
# Jalankan dengan:
#   streamlit run streamlit_app.py
# Pastikan file model 'waste_classification_mobilenetv2.h5'
# berada di direktori yang sama atau ganti path di fungsi load_model().
# -------------------------------------------------------------

IMG_SIZE = (224, 224)
CLASS_NAMES = ["organik", "anorganik", "b3"]

@st.cache_resource(show_spinner="Memuat model...")
def load_model(path: str = "waste_classification_mobilenetv2.h5"):
    """Load TensorFlow model sekali saja dan cache di memori."""
    model = tf.keras.models.load_model(path)
    return model

model = load_model()

# --------------------------- UI --------------------------------
st.set_page_config(page_title="Klasifikasi Sampah", page_icon="🗑️", layout="centered")

st.title("🗑️ Klasifikasi Sampah")
st.markdown(
    "Upload foto sampah, dan model akan memprediksi apakah **Organik**, **Anorganik**, atau **B3 (Bahan Berbahaya & Beracun)**."
)

uploaded_file = st.file_uploader("Pilih gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang di‑upload", use_column_width=True)

    # Pre‑process
    img_resized = image.resize(IMG_SIZE)
    x = np.array(img_resized) / 255.0  # normalisasi ke 0‑1
    preds = model.predict(x[np.newaxis, ...])[0]  # shape (3,)

    # Visual hasil
    pred_idx = int(np.argmax(preds))
    st.subheader(f"Prediksi: :green[**{CLASS_NAMES[pred_idx]}**]")

    prob_df = pd.DataFrame({
        "Jenis": CLASS_NAMES,
        "Probabilitas": preds
    })
    st.bar_chart(prob_df.set_index("Jenis"))

    with st.expander("Detail Probabilitas"):
        st.write({cls: f"{p * 100:.2f}%" for cls, p in zip(CLASS_NAMES, preds)})

    st.markdown("---")
    st.caption("Model: MobileNetV2 (transfer learning) – Trained on Garbage Classification dataset ➜ 3 kelas.")
