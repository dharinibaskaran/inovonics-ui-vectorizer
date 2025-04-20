import streamlit as st
import json
import time
from PIL import Image
import os
import sys
import shutil

# ==================================
# SETUP
# ==================================

print("🚀 Streamlit App Starting...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# build path to public/logo.png
logo_path = os.path.join(BASE_DIR, "public", "logo.png")
# Set page config
st.set_page_config(page_title="2D Floorplan Vectorizer", layout="wide")

# Setup Paths
UPLOAD_DIR = "./uploads/"
MODEL_DIR = "./inovonics-2D-vectorizer/models/rcnn/"
WORKING_DIR = "./inovonics-2D-vectorizer/models/rcnn/westmoor_check/"
JSON_DIR = "./inovonics-2D-vectorizer/"

# Make folders if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)
print("✅ Upload and working directories verified.")

# Import rcnn_run.py
sys.path.append(MODEL_DIR)
from rcnn_run import main, write_config

# ==================================
# CSS Styling
# ==================================

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

# ==================================
# HEADER
# ==================================

st.image(logo_path, width=250)
st.markdown("<div class='header-title'>2D Floorplan Vectorizer</div>", unsafe_allow_html=True)

# ==================================
# FILE UPLOAD SECTION
# ==================================

st.subheader("Upload your Floorplan Image")
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

# Initialize session state
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "json_output" not in st.session_state:
    st.session_state.json_output = None

# ==================================
# IMAGE + JSON Layout
# ==================================

col1, col2 = st.columns([1, 2])

# ==================================
# MAIN LOGIC
# ==================================

if uploaded_file is not None:
    print("📤 File Uploaded:", uploaded_file.name)

    # Save the uploaded file to uploads/
    # uploaded_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    # with open(uploaded_path, "wb") as f:
    #     f.write(uploaded_file.getbuffer())
    # print("✅ Uploaded file saved at:", uploaded_path)

    # # Copy the uploaded file to workingRoot
    # working_input_path = os.path.join(WORKING_DIR, uploaded_file.name)
    # shutil.copy(uploaded_path, working_input_path)
    # print("✅ File copied to model working directory:", working_input_path)


    uploaded_path = os.path.join(BASE_DIR, "uploads", uploaded_file.name)
    with open(uploaded_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    print("✅ Uploaded file saved at:", uploaded_path)

    # Copy the uploaded file to workingRoot where rcnn_run expects it
    working_input_path = os.path.join(BASE_DIR, "inovonics-2D-vectorizer", "models", "rcnn", "westmoor_check", uploaded_file.name)
    shutil.copy(uploaded_path, working_input_path)
    print("✅ File copied to model working directory:", working_input_path)


    with col1:
        st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
        st.image(Image.open(uploaded_path), caption="Uploaded Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if not st.session_state.processing_complete:
            status_placeholder = st.empty()
            status_placeholder.info("⏳ Model is processing the uploaded image...")
            # st.info("⏳ Model is processing the uploaded image...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # === 🔥 Run Model Here ===
            input_image = uploaded_file.name
            output_json = input_image.replace(".png", "_result.json").replace(".jpg", "_result.json").replace(".jpeg", "_result.json")
            output_image = input_image.replace(".png", "_result.png").replace(".jpg", "_result.png").replace(".jpeg", "_result.png")

            cfg = write_config()
            print("⚙️ Model config created. Running model...")

            # Run the model
            for i in range(1, 30):
                time.sleep(0.01)
                progress_bar.progress(i)
                status_text.text(f"Preprocessing: {i}%")
            main(cfg, input_image, output_json, output_image)
            print("✅ Model run complete.")

            # Confirm outputs exist
            output_json_path = os.path.join(BASE_DIR, "inovonics-2D-vectorizer", output_json)
            print(output_json_path)
            output_image_path = os.path.join(WORKING_DIR, output_image)

            while not os.path.exists(output_json_path):
                print("Waiting for JSON output...")
                time.sleep(0.5)

            for i in range(30, 100):
                time.sleep(0.01)
                progress_bar.progress(i)
                status_text.text(f"Postprocessing: {i}%")

            progress_bar.empty()
            status_text.text("✅ Processing Complete!")
            status_placeholder.success("✅ Model finished and JSON is ready!")


            # Read the generated JSON
            if os.path.exists(output_json_path):
                with open(output_json_path, "r") as jf:
                    st.session_state.json_output = json.load(jf)
                    print("📄 JSON Output Loaded Successfully.")
            else:
                st.session_state.json_output = {"error": "JSON output not generated."}
                print("❌ JSON output missing.")

            st.session_state.processing_complete = True

        # Display JSON
        st.markdown("<div class='json-container'>", unsafe_allow_html=True)
        st.json(st.session_state.json_output)
        st.markdown("</div>", unsafe_allow_html=True)

        # Download button
        json_str = json.dumps(st.session_state.json_output, indent=4)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="floorplan_output.json",
            mime="application/json"
        )
else:
    st.warning("⚠️ No image uploaded yet.")
    st.session_state.processing_complete = False
