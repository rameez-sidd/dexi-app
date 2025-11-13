from flask import Flask, request, send_file
import cv2
import numpy as np
import io

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['file']
    
    # Readin the raw bytes of image and converting them into a NumPy array of unsigned 8-bit integers
    image_bytes = np.frombuffer(file.read(), np.uint8)
    
    # Decoding those bytes into an actual image matrix (BGR format) using  OpenCV - basically it becomes a usable image
    img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    # Converts the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Applies Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Runs the Canny Edge-Detection Algorithm
    edges = cv2.Canny(blurred, 50, 150)
    
    # Convets grayscale image back to BGR format
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Encodes the processed image back to JPEG format - stored in memory as bytes
    _, buffer = cv2.imencode('.jpg', edges_colored)
    
    # This creates an in-memory file from those bytes and sends it back as a response to the client
    return send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype='image/jpeg'            # tells the browser that the response is an image.
    )

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host='0.0.0.0', port=port)
