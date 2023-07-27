from ultralytics import YOLO
import streamlit as st
import base64
from PIL import Image
import io
import cv2
import torch
import numpy as np

def main():
    st.title("Snapdetect")

    # File uploader to get the image from the user
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image as bytes
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # Get image dimensions and calculate display dimensions maintaining aspect ratio
        height, width = img_array.shape[:2]
        display_width = 780  # Reduced a bit to prevent cut-offs
        display_height = int((display_width / width) * height)

        # Load custom YOLO model named "best.pt"
        model = YOLO('best.pt')

        # Make predictions using the uploaded image
        results = model(img_array)
        result = results[0]  # Extract the first result

        # Process the result and draw bounding boxes and labels
        boxes = result.boxes
        for (x1, y1, x2, y2, conf, class_num) in boxes.data:
            label = result.names[int(class_num)]
            color = [int(c) for c in COLORS[int(class_num) % len(COLORS)]]
            cv2.rectangle(img_array, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.putText(img_array, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)  # Increased font size

        # Convert the image array to bytes
        is_success, img_buffer = cv2.imencode(".png", img_array)
        img_bytes = img_buffer.tobytes() if is_success else None

        # Custom HTML template for click events
        click_html = f"""
        <div style="position: relative; display: inline-block;">
            <img src="data:image/png;base64,{base64.b64encode(img_bytes).decode()}" alt="Image" width="{display_width}" height="{display_height}" onclick="handleClick(event)" onmousemove="handleMouseOver(event)">
        </div>
        <script>
            var descriptions = {{}};
            function handleClick(event) {{
                var rect = event.target.getBoundingClientRect();
                var mouseX = event.clientX - rect.left;
                var mouseY = event.clientY - rect.top;
                var description = prompt("Enter a price for this item:");
                if (description) {{
                    descriptions[mouseX + '-' + mouseY] = description;
                }}
            }}
            function handleMouseOver(event) {{
                var rect = event.target.getBoundingClientRect();
                var mouseX = event.clientX - rect.left;
                var mouseY = event.clientY - rect.top;
                var messageDiv = document.getElementById('message');
                messageDiv.innerHTML = '';
                for (var key in descriptions) {{
                    var coords = key.split('-');
                    var distance = Math.sqrt((mouseX - parseFloat(coords[0])) ** 2 + (mouseY - parseFloat(coords[1])) ** 2);
                    if (distance <= 50) {{
                        messageDiv.innerHTML += '<p style="position: absolute; left: ' + (rect.left + parseFloat(coords[0])) + 'px; top: ' + (rect.top + parseFloat(coords[1])) + 'px; background-color: #555; color: #fff; border-radius: 6px; padding: 5px;">' + descriptions[key] + '</p>';
                    }}
                }}
            }}
        </script>
        <div id="message" style="position: absolute; top: 0; left: 0;"></div>
        """

        st.components.v1.html(click_html, height=display_height, scrolling=False)

# Muted colors for bounding boxes
COLORS = [(100, 100, 100), (150, 75, 0), (75, 0, 130), (0, 75, 75), (0, 100, 0), (100, 0, 0), (50, 50, 50), (0, 50, 50), (50, 0, 50)]

if __name__ == "__main__":
    main()
