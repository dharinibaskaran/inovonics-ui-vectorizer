# 2D Floorplan Vectorizer

A Streamlit web app that allows you to upload 2D floorplan images and automatically vectorize them into COCO-style annotations using a trained Mask R-CNN model.

---

## How to Run the App

1. **Clone the repository:**

    ```bash
    git clone <this-repo-link>
    cd inovonics-ui-vectorizer
    ```

2. **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download the pretrained model:**

    - Download `model_final.pth` from [Google Drive here](https://drive.google.com/file/d/1yr64AOgaYZPTcQzG6cxG6lWBENHR9qjW/view?usp=sharing).
    - Place it inside:

      ```plaintext
      inovonics-ui-vectorizer/rcnn_model/output/model_final.pth
      ```

4. **Run the app:**

    ```bash
    streamlit run app.py
    ```

5. Open your browser at [http://localhost:8501](http://localhost:8501) to start using the app!

---

## Project Structure

```plaintext
inovonics-ui-vectorizer/
├── app.py                     # Streamlit frontend app
├── public/
│   └── logo.png                # App logo
├── rcnn_model/
│   ├── extraction/             # Extract information from uploaded png image
│   │   └── annotation_builder.py      
│   │   └── floorplan_sampler.py
│   │   └── from_labelme_runner.py
│   │   └── svg_to_json.py   
│   ├── output/                 # Empty folder while cloning. Place the pth file here
│   ├── preprocessing/          # Preprocess the image before sending to model
│   │   └── cleaning_images.py  
│   │   └── cleaning_single_image.py 
│   │   └── splitting_dataset.py
│   │   └── svg_to_yolo.py    
│   ├── results/                # Empty folder while cloning. The resulting image and JSON will be stored here
│   ├── sample/                 # Sample images for the model       
│   ├── scripts/                # Model training, evaluation and inference. Streamlit runs the rcnn_run.py file from the frontend
│   │   └── rcnn_config.py    
│   │   └── rcnn_eval.py  
│   │   └── rcnn_full_tuner.py 
│   │   └── rcnn_run.py  
│   │   └── rcnn_train.py     
│   ├── uploads/                # Temporary folder for streamlit to store the user uploaded image
│   ├── utils/                  # Utility functions during model train and preprocessing
│   │   └── coco_to_inovonics_json.py
│   │   └── floorplan_vectorizer_utils.py
│   │   └── inovonics_ann_builder.py
├── README.md                   # (this file)
├── requirements.txt            # Python dependencies
└── .gitignore                  # Files to ignore during Git commits
