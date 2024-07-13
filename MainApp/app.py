from flask import Flask, request, jsonify, render_template
import os
import cv2
import asyncio
from werkzeug.utils import secure_filename
from init_config import initialize_logging, configure_tesseract
from utilities import group_contours_by_height
from main_processing import find_rectangular_contours_with_fixed_threshold, process_image_for_all_players

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

@app.route('/', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Run the main processing function
            result = process_image(file_path)

            # Return result as JSON
            return jsonify({"result": result}), 200

        return jsonify({"error": "Invalid input"}), 400
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/result', methods=['GET'])
def show_result():
    result_data = request.args.get('result_data', {})
    return render_template('result.html', result=result_data)

def process_image(image_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(
        process_image_for_all_players(image_path, logger))
    loop.close()
    return result

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
