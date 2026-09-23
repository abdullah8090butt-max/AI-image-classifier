import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# Custom CSS
# ==========================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        margin-top: 15px;
    }

    .prediction-text {
        font-size: 30px;
        font-weight: 700;
        text-align: center;
    }

    .confidence-text {
        font-size: 22px;
        text-align: center;
    }

    .footer {
        text-align: center;
        margin-top: 40px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# Project Root
# ==========================================

project_root = os.path.dirname(
    os.path.abspath(__file__)
)


# ==========================================
# Model Path
# ==========================================

model_path = os.path.join(
    project_root,
    "cnn_model.keras"
)


# ==========================================
# CIFAR-10 Classes
# ==========================================

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ==========================================
# Load Model
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        model_path
    )


try:

    model = load_model()

except Exception as error:

    st.error(
        "Unable to load the CNN model."
    )

    st.exception(error)

    st.stop()


# ==========================================
# Sidebar
# ==========================================

with st.sidebar:

    st.header("🧠 Model Information")

    st.write("**Model:** CNN")

    st.write("**Dataset:** CIFAR-10")

    st.write("**Input Size:** 32 × 32 × 3")

    st.write("**Test Accuracy:** 68.82%")

    st.divider()

    st.subheader("Supported Classes")

    for class_name in class_names:

        st.write(
            f"• {class_name.title()}"
        )

    st.divider()

    st.caption(
        "Advanced Image Classification Project"
    )


# ==========================================
# Main Header
# ==========================================

st.markdown(
    '<div class="main-title">'
    '🧠 AI Image Classifier'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'CIFAR-10 powered image classification using a CNN model.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# Model Status
# ==========================================

st.success(
    "✅ CNN model loaded successfully"
)


# ==========================================
# Upload Section
# ==========================================

st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, or PNG image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# No Image Uploaded
# ==========================================

if uploaded_file is None:

    st.info(
        "Upload an image to begin classification."
    )

    st.markdown(
        """
        ### How it works

        **1. Upload an image**

        **2. The image is resized to 32×32**

        **3. Pixel values are normalized**

        **4. The CNN predicts the class**

        **5. Prediction probabilities are displayed**
        """
    )


# ==========================================
# Image Uploaded
# ==========================================

else:

    try:

        # ==========================================
        # Open Image
        # ==========================================

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # ==========================================
        # Create Two Columns
        # ==========================================

        image_column, result_column = st.columns(
            [1, 1]
        )


        # ==========================================
        # Display Uploaded Image
        # ==========================================

        with image_column:

            st.subheader("🖼️ Uploaded Image")

            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )

            st.caption(
                f"Original size: "
                f"{image.size[0]} × "
                f"{image.size[1]} pixels"
            )


        # ==========================================
        # Preprocessing
        # ==========================================

        image_resized = image.resize(
            (32, 32)
        )

        image_array = np.array(
            image_resized
        ).astype("float32")

        image_array = (
            image_array / 255.0
        )

        image_input = np.expand_dims(
            image_array,
            axis=0
        )


        # ==========================================
        # Prediction
        # ==========================================

        with st.spinner(
            "Analyzing image..."
        ):

            prediction = model.predict(
                image_input,
                verbose=0
            )


        predicted_index = np.argmax(
            prediction[0]
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = (
            float(
                prediction[0][predicted_index]
            ) * 100
        )


        # ==========================================
        # Prediction Result
        # ==========================================

        with result_column:

            st.subheader("🎯 Prediction")

            st.markdown(
                '<div class="result-box">',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="prediction-text">'
                f'{predicted_class.title()}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="confidence-text">'
                f'{confidence:.2f}% confidence'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


            # ==========================================
            # Confidence Information
            # ==========================================

            if confidence >= 70:

                st.success(
                    "The model shows relatively high "
                    "confidence in this prediction."
                )

            elif confidence >= 40:

                st.warning(
                    "The model shows moderate confidence "
                    "in this prediction."
                )

            else:

                st.warning(
                    "The model is uncertain about this image."
                )


        # ==========================================
        # Class Probabilities
        # ==========================================

        st.divider()

        st.subheader(
            "📊 Class Probabilities"
        )

        probabilities = (
            prediction[0] * 100
        )

        sorted_indices = np.argsort(
            probabilities
        )[::-1]


        for index in sorted_indices:

            class_name = class_names[
                index
            ]

            probability = float(
                probabilities[index]
            )

            st.write(
                f"**{class_name.title()}** — "
                f"{probability:.2f}%"
            )

            st.progress(
                min(
                    probability / 100,
                    1.0
                )
            )


        # ==========================================
        # Technical Details
        # ==========================================

        with st.expander(
            "🔧 Technical Details"
        ):

            st.write(
                "Model input shape:",
                image_input.shape
            )

            st.write(
                "Resized image:",
                image_resized.size
            )

            st.write(
                "Pixel range:",
                f"{image_input.min():.2f} "
                f"to {image_input.max():.2f}"
            )

            st.write(
                "Number of classes:",
                len(class_names)
            )


    except Exception as error:

        st.error(
            "❌ Something went wrong while "
            "processing the image."
        )

        st.exception(error)


# ==========================================
# Footer
# ==========================================

st.markdown(
    '<div class="footer">'
    'Built with TensorFlow + Streamlit | '
    'CIFAR-10 CNN Image Classification'
    '</div>',
    unsafe_allow_html=True
)