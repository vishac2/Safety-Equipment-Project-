import streamlit as st
import os
import tempfile
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2

# Set page configuration
st.set_page_config(
    page_title="Safety Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS to improve appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 20px;
        text-align: center;
    }
    .subtitle {
        font-size: 24px;
        margin-bottom: 20px;
        color: #3B82F6;
        text-align: center;
    }
    .stButton > button {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
        width: 100%;
    }
    .stProgress > div > div > div {
        background-color: #2563EB;
    }
    .detection-stats {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# App header
st.markdown("<div class='title'>Safety Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image for safety equipment detection</div>", unsafe_allow_html=True)

# Function to load the model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Model path
model_path = st.sidebar.text_input(
    "Model Path", 
    value="best.pt", 
    help="Enter the path to your trained YOLO model"
)

# Load model
try:
    with st.spinner("Loading YOLO model..."):
        model = load_model(model_path)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    model = None

# Confidence threshold slider
conf_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05,
)

# Create two columns for input and output
col1, col2 = st.columns(2)

# File uploader
with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name

        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

# Process and display results
with col2:
    st.subheader("Detection Results")
    
    if uploaded_file is not None and model is not None:
        if st.button("Run Detection"):
            with st.spinner("Processing..."):
                # Run the model on the uploaded image
                results = model.predict(
                    source=temp_file_path,
                    conf=conf_threshold,
                    save=True
                )

                # Display detected image
                for r in results:
                    # Get the plotting array
                    im_array = r.plot()
                    im = Image.fromarray(im_array[..., ::-1])  # RGB to BGR
                    st.image(im, caption="Detected Objects", use_column_width=True)
                    
                    # Extract and display detection statistics
                    boxes = r.boxes
                    
                    if len(boxes) > 0:
                        # Create dataframe for detections
                        detections_data = []
                        class_names = model.names
                        
                        # Count detections by class
                        class_counts = {}
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            cls_name = class_names[cls_id]
                            conf = box.conf[0].item()
                            
                            if cls_name in class_counts:
                                class_counts[cls_name] += 1
                            else:
                                class_counts[cls_name] = 1
                                
                            # Add detection to data list
                            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0].tolist()]
                            detections_data.append({
                                "Class": cls_name,
                                "Confidence": f"{conf:.2f}",
                                "Coordinates": f"({x1}, {y1}, {x2}, {y2})"
                            })
                            
                        # Display statistics
                        st.markdown("<div class='detection-stats'>", unsafe_allow_html=True)
                        st.subheader("Detection Statistics")
                        
                        # Total detections count
                        total_detections = len(boxes)
                        st.markdown(f"**Total detections:** {total_detections}")
                        
                        # Class distribution
                        st.markdown("**Class distribution:**")
                        
                        # Create a horizontal bar chart for class distribution
                        if class_counts:
                            fig, ax = plt.subplots(figsize=(8, max(3, len(class_counts) * 0.5)))
                            classes = list(class_counts.keys())
                            counts = list(class_counts.values())
                            
                            y_pos = np.arange(len(classes))
                            ax.barh(y_pos, counts, align='center')
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(classes)
                            ax.invert_yaxis()  # Labels read top-to-bottom
                            ax.set_xlabel('Count')
                            ax.set_title('Detection Class Distribution')
                            
                            st.pyplot(fig)
                        
                        # Show detailed detection information
                        if detections_data:
                            st.markdown("**Detailed Detections:**")
                            st.dataframe(pd.DataFrame(detections_data))
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.warning("No objects detected in the image.")
                
                # Clean up the temporary file
                os.unlink(temp_file_path)

# Add information in the sidebar
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    This application uses a YOLO model trained for safety equipment detection.
    
    **How to use:**
    1. Enter the path to your trained model
    2. Upload an image
    3. Adjust confidence threshold if needed
    4. Click 'Run Detection'
    5. View results and statistics
    """)
    
    st.markdown("## Model Information")
    st.text(f"Model: {model_path}")
    
    # Add your custom metadata about the model
    st.markdown("""
    **Safety Detection Categories:**
    - Hard Hat
    - Safety Vest
    - Safety Glasses
    - Safety Gloves
    - Safety Boots
    
    Adjust the list above based on your actual model classes.
    """)