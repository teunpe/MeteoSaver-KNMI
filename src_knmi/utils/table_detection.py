import logging
import pytesseract
import pandas as pd
import re
import cv2
import numpy as np

logger = logging.getLogger('meteosaver')

def add_missing_boxes(sorted_rows, max_cell_width_threshold=130, max_cell_height_threshold=50, max_columns=24):
    updated_rows = []

    for row in sorted_rows:
        if any(cell is None for cell in row):
            continue

        bounding_boxes = [cv2.boundingRect(c) for c in row]
        bounding_boxes.sort(key=lambda b: b[0])

        new_boxes = bounding_boxes.copy()
        gaps = []

        # Handle missing boxes at the start of the row
        if new_boxes and new_boxes[0][0] > max_cell_width_threshold:
            first_x, first_y, first_w, first_h = new_boxes[0]
            num_missing_boxes = min(int(first_x // max_cell_width_threshold), max_columns - len(new_boxes))
            new_boxes_at_start = []
            for i in range(num_missing_boxes):
                new_x = max(0, first_x - (num_missing_boxes - i) * max_cell_width_threshold*1.1)
                # print(f'x: {first_x}, missing: {num_missing_boxes}, i: {i}, threshold: {max_cell_width_threshold}')
                # print(new_x)
                new_box = (new_x, first_y, max_cell_width_threshold, max_cell_height_threshold)
                new_boxes_at_start.append(new_box)

            new_boxes = new_boxes_at_start + new_boxes

        for i in range(len(new_boxes) - 1):
            x1, y1, w1, h1 = new_boxes[i]
            x2, _, _, _ = new_boxes[i + 1]
            gap = x2 - (x1 + w1)

            if gap > max_cell_width_threshold*0.9:
                gaps.append((gap, i, x1 + w1, y1))

        gaps.sort(reverse=True, key=lambda g: g[0])

        for gap, i, gap_start_x, y1 in gaps:
            if len(new_boxes) >= max_columns:
                break

            num_missing_boxes = min(int(gap // max_cell_width_threshold), max_columns - len(new_boxes))

            total_box_width = num_missing_boxes * max_cell_width_threshold
            # start_x = gap_start_x + (gap - total_box_width) / 2

            # new_boxes_in_gap = []
            # for j in range(num_missing_boxes):
            #     new_x = start_x + j * max_cell_width_threshold
            #     new_box = (int(new_x), y1, max_cell_width_threshold, max_cell_height_threshold)
            #     new_boxes_in_gap.append(new_box)
            
            # Dynamically compute width so that all boxes fit perfectly into the gap
            dynamic_cell_width = gap / (num_missing_boxes)
            # print(dynamic_cell_width, gap, num_missing_boxes)
            new_boxes_in_gap = []
            for j in range(num_missing_boxes):
                if len(new_boxes) + len(new_boxes_in_gap) >= max_columns:
                    break

                center_x = gap_start_x + (j+1) * dynamic_cell_width
                new_x = int(center_x - dynamic_cell_width*0.9)

                new_box = (new_x, y1, int(dynamic_cell_width)*0.8, max_cell_height_threshold)
                new_boxes_in_gap.append(new_box)
            
                # if len(new_boxes) + len(new_boxes_in_gap) >= max_columns:
                #     break

            new_boxes[i + 1:i + 1] = new_boxes_in_gap

        new_boxes = new_boxes[:max_columns]

        updated_contours = [
            np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
            for x, y, w, h in new_boxes
        ]
        updated_rows.append(updated_contours)

    return updated_rows

def find_word(image: np.ndarray, reference_word: str = "DATUM", hor_offset: int=3000, vert_offset: int=1000):
    """
    Return x, y of reference word "DATUM" in image
    """    
    word = reference_word
    cropped = image[vert_offset:vert_offset+1000, hor_offset:hor_offset+1000]
    # cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('custom window', cropped )
    # cv2.resizeWindow('custom window', 1080, 920)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # plt.imshow(cropped, cmap='gray')
    data = pytesseract.image_to_data(cropped, lang='nld', 
                                     output_type=pytesseract.Output.DICT)

    # Convert the data to a DataFrame for easier manipulation
    df = pd.DataFrame(data)
    x = None
    # Find the word and get its coordinates
    for i in range(len(df['text'])):
        if re.sub('[^a-zA-Z]+', '', str(df['text'][i]).lower().strip('."')) == word.lower():
            (x, y) = (df['left'][i]+hor_offset, df['top'][i]+vert_offset)
            print(f"The word '{word}' was found at coordinates: "\
                  f"(x={x}, y={y})")
    if x is not None:
        return x, y, df
    else:
        return None, None, df
    
def table_detection(binarized_image: np.ndarray, reference_word: str = "datum", size: tuple = (3663, 2601), 
                    initial_offset: tuple =(3000, 1000), table_offset: tuple = (150, 170)):
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
    
    (x, y, df) = find_word(binarized_image, reference_word, initial_offset[0], initial_offset[1])

    if x is None or y is None:
        logger.error("No table detected. Returning detected text.")
        return None, None, None, None, df
    w = size[0]
    h = size[1]
    x, y = (x+table_offset[0], y+table_offset[1])

    return (x, y, w, h, df)