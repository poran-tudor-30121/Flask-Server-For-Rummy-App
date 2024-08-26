import asyncio
import cv2
from init_config import initialize_logging, configure_tesseract
from utilities import group_contours_by_height
from main_processing_test import process_image_for_some_players


def main(image_path, tesseract_path):
    logger = initialize_logging()
    configure_tesseract(tesseract_path)
    loop = asyncio.get_event_loop()
    #result = loop.run_until_complete(process_image_for_all_players(image_path, logger))
    result = loop.run_until_complete(process_image_for_some_players(image_path, logger,wanted_sections=['left','bottom','top']))
    logger.info(result)

     
if __name__ == '__main__':
    image_path = 'poze/pozapejos3.jpg'
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    main(image_path, tesseract_path)
