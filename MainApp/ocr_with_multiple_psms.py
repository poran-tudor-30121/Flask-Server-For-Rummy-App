import pytesseract
import asyncio

async def ocr_with_multiple_psms(image, psms):
    # Initialize a dictionary to store OCR results and their frequencies
    ocr_results = {}

    async def ocr_with_psm(psm):
        text = await asyncio.to_thread(pytesseract.image_to_string, image, config=f'--psm {psm} -c tessedit_char_whitelist=0123456789')
        return text

    # Gather OCR results asynchronously
    ocr_tasks = [ocr_with_psm(psm) for psm in psms]
    ocr_results_list = await asyncio.gather(*ocr_tasks)

    # Process OCR results
    for text in ocr_results_list:
        if text.strip().isdigit() and int(text.strip()) <= 13:
            if text in ocr_results:
                ocr_results[text] += 1
            else:
                ocr_results[text] = 1
        if not ocr_results:
            most_common_result = "Jolly"
        else:
            most_common_result = max(ocr_results, key=ocr_results.get)


    return most_common_result
