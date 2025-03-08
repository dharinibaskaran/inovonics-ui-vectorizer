# Inovonics - Vectorizing 2D Floor Plans


**App** : https://inovonics-ui-vectorizer.streamlit.app/

**Repo** : https://github.com/dharinibaskaran/inovonics-ui-vectorizer

**Demo** : [UI Demo Recording](https://drive.google.com/file/d/1eUei_fAmglMts_uOrRgPS-scszXtKun1/view?usp=drive_link)

**Introduction**

The 2D Floorplan Vectorizer is a web-based application designed to process floorplan images and generate structured JSON data. The application allows users to upload floorplan images, processes them using an external model, and provides the extracted vectorized data in a structured format. It features a clean and minimalist user interface with enhanced usability, including a progress bar, JSON display, and a download option.

**Key Features**


**File Upload System**

Users can upload PNG, JPG, and JPEG images of floor plans.
The uploaded image is displayed for review.


**Processing Workflow**

Once an image is uploaded, a simulated progress bar provides real-time feedback on processing status.
The application is designed to integrate with an external Python script (model) that processes the image and generates a corresponding data.json file.


**JSON Display & Download**

<li>
The extracted vectorized data is displayed in a scrollable JSON viewer.
<li>
Users can download the JSON file for further use.

<br>

**User Interface Enhancements**

<li>
Minimalist UI Design: A clean interface with soft pastel color gradients.
<li>
Scrollable JSON Viewer: Ensures large JSON data remains easy to navigate.
<li>
Dynamic Progress Bar: Provides visual feedback during processing.
<li>
Custom Styling: Improved button styles, structured layout, and responsive design.

<br>

**Technology Stack**
<li>
Frontend: Streamlit (for UI and interaction)
<li>
Backend Processing: External Python script for image processing and JSON generation
<li>
Styling: Custom CSS for enhanced UI aesthetics

<br>

**Why Streamlit Over React?**
 
Streamlit was chosen over React for its simplicity, quick deployment, and seamless Python integration. Unlike React, Streamlit provides built-in UI components like file uploaders, progress bars, and JSON viewers, reducing development time. Since the project is data-driven and heavily relies on Python-based processing, Streamlit allows direct interaction with the backend model without needing a separate API layer, making it the ideal choice for this workflow.

**Conclusion**

Moving forward, the component to include the processed image will also be integrated to this UI. The 2D Floorplan Vectorizer provides an efficient solution for converting floorplan images into structured data. With its user-friendly interface and structured workflow, it simplifies the process of extracting and utilizing vectorized floorplan data for further analysis and applications. The decision to use Streamlit over React ensures ease of development, better Python integration, and faster deployment for data-intensive workflows.


