import cv2
import numpy as np
import base64  # Import base64 module
from django.shortcuts import render
from .forms import ImageUploadForm
from ultralytics import YOLO
import matplotlib.pyplot as plt
import io
from django.http import HttpResponse

# Load the YOLOv9c model
model = YOLO('yolov9c.pt')  # Ensure you have your model path correct

def enhance_image(image):
    """Enhance the image using CLAHE for better contrast."""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_image = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)
    return enhanced_image

def upload_image(request):
    annotated_image_url = None
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']
            # Read the image file
            image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

            # Enhance the image
            enhanced_image = enhance_image(image)

            # Perform object detection on the enhanced image
            results = model(enhanced_image)
            annotated_image = results[0].plot()

            # Save the annotated image to a BytesIO object
            buf = io.BytesIO()
            plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)

            # Convert image data to a base64 string for rendering
            annotated_image_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

    else:
        form = ImageUploadForm()

    return render(request, 'detection/upload.html', {'form': form, 'annotated_image_url': annotated_image_url})
