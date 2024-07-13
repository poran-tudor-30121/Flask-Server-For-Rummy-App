import logging
import pytesseract

def initialize_logging():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return logging.getLogger()

def configure_tesseract(path):
    pytesseract.pytesseract.tesseract_cmd = path
