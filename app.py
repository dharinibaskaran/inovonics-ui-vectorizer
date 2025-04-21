import streamlit as st
st.set_page_config(
    page_title="2D Floorplan Vectorizer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import json
import time
from PIL import Image
import os
import sys

# print("Streamlit App Starting...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Setup Paths
UPLOAD_DIR = os.path.join(BASE_DIR, "rcnn_model", "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "rcnn_model", "scripts")
JSON_DIR = os.path.join(BASE_DIR, "rcnn_model", "results")
SAMPLE_DIR = os.path.join(BASE_DIR, "rcnn_model", "sample")
logo_path = os.path.join(BASE_DIR, "public", "logo.png")

# Make folders if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# Import rcnn_run.py -- > this is the main model file for running predictions on single or batch images
sys.path.append(MODEL_DIR)
from rcnn_model.scripts.rcnn_run import main, write_config

st.markdown(
    """
    <style>
    .stApp { background-color: #FAFAFA; }
    .header-title { font-size: 2.5rem; font-weight: bold; text-align: center;
                    background: linear-gradient(to right, #D4ECDD, #EAF4F4);
                    color: #2C3E50; padding: 20px; border-radius: 12px; }
    .upload-container { display: flex; flex-direction: column; align-items: center;
                        justify-content: center; background: white; padding: 20px;
                        border-radius: 10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); }
    .json-container { background: #F5F5F5; padding: 15px; border-radius: 10px;
                      font-family: monospace; overflow-y: auto; max-height: 400px;
                      white-space: pre-wrap; }
    </style>
    """,
    unsafe_allow_html=True
)

st.image(logo_path, width=250)
st.markdown("<div class='header-title'>2D Floorplan Vectorizer</div>", unsafe_allow_html=True)

st.subheader("Upload your Floorplan Image")
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "json_output" not in st.session_state:
    st.session_state.json_output = None

col1, col2 = st.columns([1, 2])

if uploaded_file is not None:
    # print("File Uploaded:", uploaded_file.name)

    # Save uploaded file
    uploaded_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(uploaded_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # print("Uploaded file saved at:", uploaded_path)

    # Display uploaded image
    with col1:
        st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
        st.image(Image.open(uploaded_path), caption="Uploaded Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if not st.session_state.processing_complete:
            status_placeholder = st.empty()
            status_placeholder.info("⏳ Model is processing the uploaded image...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Running Model Here 
            input_image = uploaded_path
            output_json_name = uploaded_file.name.replace(".png", "_result.json").replace(".jpg", "_result.json").replace(".jpeg", "_result.json")
            output_image_name = uploaded_file.name.replace(".png", "_result.png").replace(".jpg", "_result.png").replace(".jpeg", "_result.png")

            cfg = write_config()
            # print("Model config created. Running model...")

            # Simulating Progress Bar
            for i in range(1, 30):
                time.sleep(0.01)
                progress_bar.progress(i)
                status_text.text(f"Preprocessing: {i}%")

            main(cfg, input_image, output_json_name, output_image_name)
            # print("Model run complete.")

            # Prepare Output Paths
            output_json_path = os.path.join(JSON_DIR, output_json_name)
            output_image_path = os.path.join(JSON_DIR, output_image_name)

            while not os.path.exists(output_json_path):
                # print("Waiting for JSON output...")
                time.sleep(0.5)

            for i in range(30, 100):
                time.sleep(0.01)
                progress_bar.progress(i)
                status_text.text(f"Postprocessing: {i}%")

            progress_bar.empty()
            status_text.text("✅ Processing Complete!")
            status_placeholder.success("✅ Model finished and JSON is ready!")

            # Read generated JSON
            if os.path.exists(output_json_path):
                with open(output_json_path, "r") as jf:
                    st.session_state.json_output = json.load(jf)
                    # print("JSON Output Loaded Successfully.")
            else:
                st.session_state.json_output = {"error": "JSON output not generated."}
                # print("JSON output missing.")

            st.session_state.processing_complete = True

        # DISPLAY Output Image and JSON file

        out_col1, out_col2 = st.columns(2)

        with out_col1:
            if os.path.exists(output_image_path):
                st.image(output_image_path, caption="🖼 Output Vectorized Image", use_container_width=True)

                with open(output_image_path, "rb") as img_file:
                    st.download_button(
                        label="Download Output Image",
                        data=img_file,
                        file_name="floorplan_output.png",
                        mime="image/png"
                    )

                json_str = json.dumps(st.session_state.json_output, indent=4)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="floorplan_output.json",
                    mime="application/json"
                )
            else:
                st.warning("⚠️ Output image not found.")

        with out_col2:
            st.markdown("<div class='json-container'>", unsafe_allow_html=True)
            st.json(st.session_state.json_output)
            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.warning("⚠️ No image uploaded yet.")
    st.session_state.processing_complete = False