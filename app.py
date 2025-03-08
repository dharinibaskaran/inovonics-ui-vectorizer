import streamlit as st
import json
import time
from PIL import Image

# Set page config
st.set_page_config(page_title="2D Floorplan Vectorizer", layout="wide")

# CSS styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FAFAFA; /* Off-White */
    }
.header-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(to right, #D4ECDD, #EAF4F4); /* Soft Pastel Green to Sage */
color: #2C3E50; /* Dark Blue-Grey */
        padding: 20px;
        border-radius: 12px;
    }
    .upload-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .image-box {
        display: flex;
        align-items: center;
        justify-content: center;
        background: white;
        padding: 20px;
        border-radius: 10px;
    }
    .json-container {
        background: #F5F5F5; /* Very Light Grey */
        padding: 15px;
        border-radius: 10px;
        font-family: monospace;
        overflow-y: auto;
        max-height: 400px; /* Limit the height */
        white-space: pre-wrap;
        display: block;
    }
.stButton>button {
        background-color: #0288D1 !important; /* Bright Sky Blue */
        color: white !important;
        border-radius: 5px !important;
        font-size: 16px !important;
    }
    .stButton>button:hover {
        background-color: #424242 !important;
    }
    .stAlert {
    background-color: #FFF3E0 !important; /* Soft Orange Background */
    color: #BF360C !important; /* Dark Orange Text */
    border-radius: 8px;
    font-weight: bold;
}
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.image("./public/logo.png", width=250)
st.markdown("<div class='header-title'>2D Floorplan Vectorizer</div>", unsafe_allow_html=True)

# File Upload Section
st.subheader("Upload your Floorplan Image")
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

# Initialize session state for processing tracking
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "json_output" not in st.session_state:
    st.session_state.json_output = None

# Image and JSON Output Layout
col1, col2 = st.columns([1, 2])

# Handle Image Upload
if uploaded_file is not None:
    with col1:
        st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
        st.image(Image.open(uploaded_file), caption="Uploaded Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        if not st.session_state.processing_complete:
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(1, 101):  # Simulate processing steps
                time.sleep(0.05)  # Adjust speed of progress
                progress_bar.progress(i)
                status_text.text(f"Processing: {i}%")

            progress_bar.empty()
            status_text.text("Processing Complete!")
            
            # Load JSON output from data.json
            try:
                with open("data.json", "r") as json_file:
                    st.session_state.json_output = json.load(json_file)
            except Exception as e:
                st.session_state.json_output = {"error": f"Failed to load JSON: {str(e)}"}
            
            # Mark processing as complete
            st.session_state.processing_complete = True

        # Display JSON output
        st.markdown("<div class='json-container'>", unsafe_allow_html=True)
        st.json(st.session_state.json_output)
        st.markdown("</div>", unsafe_allow_html=True)

        # Provide Download JSON Button
        json_str = json.dumps(st.session_state.json_output, indent=4)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="output.json",
            mime="application/json"
        )

else:
    st.warning("No image uploaded yet.")
    st.session_state.processing_complete = False  # Reset when no file is uploaded
