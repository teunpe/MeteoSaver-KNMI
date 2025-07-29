import logging
import pytesseract
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

logger = logging.getLogger('meteosaver')

def find_word(image: np.ndarray):
    """
    Return x, y of reference word "DATUM" in image
    """    
    word = 'DATUM.'
    cropped = image[:1000, :1500]
    
    data = pytesseract.image_to_data(cropped, lang='nld', 
                                     output_type=pytesseract.Output.DICT)

    # Convert the data to a DataFrame for easier manipulation
    df = pd.DataFrame(data)
    x = None
    # Find the word and get its coordinates
    for i in range(len(df['text'])):
        if df['text'][i] == word:
            (x, y) = (df['left'][i], df['top'][i])
            print(f"The word '{word}' was found at coordinates: "\
                  "(x={x}, y={y})")
    if x is not None:
        return x, y
    else:
        return None, None
    
def table_detection(binarized_image: np.ndarray):
    """Returns the coordinates of the table in the image by finding the 
    word "DATUM" in the image and using default values for the table 
    width and height.

    Parameters
    ----------
    binarized_image
        
    Returns
    -------
        (x, y, w, h): Coordinates of top left corner of the table and 
        its width and height
    """    
    (x, y) = find_word(binarized_image)

    if x is None or y is None:
        logger.error("No table detected. Returning None.")
        return None
    w = 3663
    h = 2262
    x, y = (x+150, y+170)

    return (x, y, w, h)
