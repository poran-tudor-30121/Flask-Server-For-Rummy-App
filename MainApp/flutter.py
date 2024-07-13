from flask import Flask, request, jsonify, make_response
import os
import cv2
import asyncio
from werkzeug.utils import secure_filename
from init_config import initialize_logging, configure_tesseract
from main_processing import find_rectangular_contours_with_fixed_threshold

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Configure the logger
logger = initialize_logging()

# Set tesseract path (adjust according to your installation)
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
configure_tesseract(TESSERACT_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.errorhandler(500)
def internal_error(error):
    response = jsonify({"error": "Internal Server Error"})
    response.status_code = 500
    return response

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        target_num_rectangles = request.form.get('target_num_rectangles')
        if file and allowed_file(file.filename) and target_num_rectangles.isdigit():
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Run the main processing function
            target_num_rectangles = int(target_num_rectangles)
            result = process_image(file_path, target_num_rectangles)

            return jsonify({"result": result})

        return jsonify({"error": "Invalid input"}), 400
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

def process_image(image_path, target_num_rectangles):
    image = cv2.imread(image_path)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(
        find_rectangular_contours_with_fixed_threshold(image, target_num_rectangles, logger))
    loop.close()
    return result

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
