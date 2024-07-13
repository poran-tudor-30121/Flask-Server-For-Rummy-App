import pytesseract

def ocr_with_multiple_psms_first(image, psms):
    # Initialize a dictionary to store OCR results and their frequencies
    ocr_results = {}

    # Iterate over each specified PSM
    for psm in psms:
        # Perform OCR using the current PSM
        text = pytesseract.image_to_string(image, config=f'--psm {psm} -c tessedit_char_whitelist=0123456789')
        print("PSM-ul ", psm, " VEDE:", text)
        # Check if the OCR result is a digit between 1 and 13
        if text.strip().isdigit() and int(text.strip()) <= 13:
            # Update the dictionary with the OCR result and its frequency
            if text in ocr_results:
                ocr_results[text] += 1
            else:
                ocr_results[text] = 1

    # Check if both "4" and "41" are present in the OCR results
    if (ocr_results.get('4\n', 0) > 0 and ocr_results.get('41\n', 0) > 0) or (ocr_results.get('1\n', 0) > 0 and ocr_results.get('4\n', 0) > 0):
        most_common_result = "1"
    else:
        # If no valid digit results are found or only one of "4" or "41" is present,
        # set the most common result to the key with the highest frequency
        if not ocr_results:
            most_common_result = "Jolly"
        else:
            most_common_result = max(ocr_results, key=ocr_results.get)

    return most_common_result