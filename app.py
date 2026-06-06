import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Exercise Posture Classification",
    page_icon="🏋️",
    layout="centered"
)

st.title("🏋️ Exercise Posture Classification System")
st.write("Upload an exercise image to predict the exercise posture.")

MODEL_PATH = "exercise_posture_custom_cnn.keras"
CLASS_PATH = "class_names.json"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Model loaded successfully.")
    st.write("Model input shape:", model.input_shape)
    st.write("Model output shape:", model.output_shape)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

try:
    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)

    st.write("Number of classes:", len(class_names))
except Exception as e:
    st.error(f"Class names loading failed: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload exercise image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        input_height = model.input_shape[1]
        input_width = model.input_shape[2]

        image_resized = image.resize((input_width, input_height))

        image_array = np.array(image_resized).astype("float32")
        image_array = np.expand_dims(image_array, axis=0)

        st.write("Image array shape:", image_array.shape)

        try:
            with st.spinner("Predicting... please wait"):
                prediction = model(image_array, training=False).numpy()

            st.write("Prediction completed")
            st.write("Prediction shape:", prediction.shape)

            predicted_index = int(np.argmax(prediction[0]))
            predicted_class = class_names[predicted_index]
            confidence = float(np.max(prediction[0]) * 100)

            st.subheader("Prediction Result")
            st.success(f"Predicted Exercise: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")

            top_indices = np.argsort(prediction[0])[-5:][::-1]

            result_df = pd.DataFrame({
                "Exercise": [class_names[i] for i in top_indices],
                "Confidence (%)": [prediction[0][i] * 100 for i in top_indices]
            })

            st.subheader("Top 5 Predictions")
            st.dataframe(result_df, use_container_width=True)
            st.bar_chart(result_df.set_index("Exercise"))

        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.warning("Please upload an image.")
