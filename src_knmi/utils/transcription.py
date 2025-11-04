
from sklearn.cluster import KMeans
import numpy as np
import cv2
import zipfile
import os
import pytesseract
import easyocr
from tqdm.auto import tqdm
from src.transcription import organize_contours_midpoint, organize_contours_top, get_max_rows_from_filename#, add_missing_boxes, 
from src.transcription import generate_random_colors, draw_row_markers_and_boxes, calculate_trimmed_mean
from src.transcription import calculate_cell_reference
from openpyxl.styles import Border, Side, Alignment
import openpyxl
from openpyxl import Workbook
from openpyxl.utils import get_column_letter


import gc
import time
import psutil
import tracemalloc

proc = psutil.Process(os.getpid())
def log_mem(tag):
    rss = proc.memory_info().rss / 1024**2
    try:
        top = tracemalloc.take_snapshot().statistics('lineno')[0]
        trac = f"tracemalloc top: {top.size/1024:.1f} KiB @ {top.traceback}" 
    except Exception:
        trac = ""
    print(f"[MEM] {tag} RSS={rss:.1f} MiB {trac}")

# start tracemalloc early (optional, small cost)
tracemalloc.start()


def organize_columns(contours, filename, max_cols=None):
    if not contours:
        return []

    # Step 1: Extract **left edge x-coordinates** for clustering
    left_edges = np.array([cv2.boundingRect(c)[0] for c in contours]).reshape(-1, 1)
    
    # Step 2: Perform K-Means clustering on top edges
    if max_cols is None:
        return []
        # max_cols = get_max_cols_from_filename(filename)
    kmeans = KMeans(n_clusters=min(max_cols, len(left_edges)), random_state=0, n_init=50, tol=1e-2)
    kmeans.fit(left_edges)
    labels = kmeans.labels_

    # Step 3: Assign contours to rows based on clustering
    col_dict = {i: [] for i in range(max_cols)}
    for label, contour in zip(labels, contours):
        col_dict[label].append(contour)

    # Step 4: Compute **super strict col medians** using only the closest 50%
    col_medians = {}

    for i, col in col_dict.items():
        if len(col) > 2:
            x_lefts = np.array([cv2.boundingRect(c)[0] for c in col])
            x_lefts.sort()

            # Keep **only the middle 50% closest values**
            trimmed_x_lefts = x_lefts[len(x_lefts) // 4 : 3 * len(x_lefts) // 4]

            # Compute median of the **trimmed** values
            col_medians[i] = np.median(trimmed_x_lefts)
        else:
            col_medians[i] = np.median([cv2.boundingRect(c)[0] for c in col])

    # Step 5: Move misplaced boxes to the best col
    adjusted_cols = {i: [] for i in range(max_cols)}

    for i, col in col_dict.items():
        for box in col:
            x_left = cv2.boundingRect(box)[0]  # **Left edge x-coordinate**

            # Find closest **trusted** col median (ignoring outliers)
            closest_col = i
            min_distance = abs(x_left - col_medians[0])

            if i > 0:  # Check col to the left
                distance_left = abs(x_left - col_medians[i-1])
                if distance_left < min_distance:
                    min_distance = distance_left
                    closest_col = i-1

            if i < max_cols - 1:  # Check col to the right
                distance_right = abs(x_left - col_medians[i+1])
                if distance_right < min_distance:
                    closest_col = i+1

            adjusted_cols[closest_col].append(box)

    # Step 6: Sort each adjusted col again (top to bottom)
    for i in range(len(adjusted_cols)):
        adjusted_cols[i] = sorted(adjusted_cols[i], key=lambda c: cv2.boundingRect(c)[1])

    # Convert to final sorted list
    sorted_cols = [adjusted_cols[i] for i in range(max_cols)]

    return sorted_cols

def remove_small_cells_in_column(sorted_columns):
    for col in sorted_columns:
        if not col:
            continue

        heights = [cv2.boundingRect(c)[3] for c in col]
        median_height = np.median(heights)

        widths = [cv2.boundingRect(c)[2] for c in col]
        median_width = np.median(widths)

        # print(cv2.boundingRect(col[0]))
        filtered_col = [c for c in col if cv2.boundingRect(c)[3] >= 0.2 * median_height and cv2.boundingRect(c)[2] >= 0.6 * median_width]
        # print(f'Removed {len(col) - len(filtered_col)} small cells from a column of {len(col)} cells.')
        col.clear()
        col.extend(filtered_col)
    return sorted_columns

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
            if num_missing_boxes > 0:
                
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

                    new_box = (new_x, y1, int(dynamic_cell_width)*0.9, max_cell_height_threshold)
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

def sort_columns_to_rows(sorted_columns, max_rows):
    
    # Step 1: Extract **top edge y-coordinates** for clustering
    top_edges = np.array([cv2.boundingRect(c)[1] for col in sorted_columns for c in col]).reshape(-1, 1)
    # print(top_edges)
    # Step 2: Perform K-Means clustering on top edges
    # if max_rows is None:
    #     max_rows = get_max_rows_from_filename(filename)
    kmeans = KMeans(n_clusters=min(max_rows, len(top_edges)), random_state=0, n_init=50, tol=1e-2)
    kmeans.fit(top_edges)
    labels = kmeans.labels_

    #get contours from sorted columns
    contours = [c for col in sorted_columns for c in col if c is not None]

    # Step 3: Assign contours to rows based on clustering
    row_dict = {i: [] for i in range(max_rows)}
    for label, contour in zip(labels, contours):
        row_dict[label].append(contour)

    # Step 4: Compute **super strict row medians** using only the closest 50%
    row_medians = {}

    for i, row in row_dict.items():
        if len(row) > 2:
            y_tops = np.array([cv2.boundingRect(c)[1] for c in row])
            y_tops.sort()

            # Keep **only the middle 50% closest values**
            trimmed_y_tops = y_tops[len(y_tops) // 4 : 3 * len(y_tops) // 4]

            # Compute median of the **trimmed** values
            row_medians[i] = np.median(trimmed_y_tops)
        else:
            row_medians[i] = np.median([cv2.boundingRect(c)[1] for c in row])

    # Step 5: Move misplaced boxes to the best row
    adjusted_rows = {i: [] for i in range(max_rows)}

    for i, row in row_dict.items():
        for box in row:
            y_top = cv2.boundingRect(box)[1]  # **Top edge y-coordinate**
            
            # Find closest **trusted** row median (ignoring outliers)
            closest_row = i
            min_distance = abs(y_top - row_medians[i])

            if i > 0:  # Check row above
                distance_up = abs(y_top - row_medians[i-1])
                if distance_up < min_distance:
                    min_distance = distance_up
                    closest_row = i-1

            if i < max_rows - 1:  # Check row below
                distance_down = abs(y_top - row_medians[i+1])
                if distance_down < min_distance:
                    closest_row = i+1

            adjusted_rows[closest_row].append(box)

    # Step 6: Sort each adjusted row again (left to right)
    for i in range(len(adjusted_rows)):
        adjusted_rows[i] = sorted(adjusted_rows[i], key=lambda c: cv2.boundingRect(c)[0])

    # Convert to final sorted list
    sorted_rows = [adjusted_rows[i] for i in range(max_rows)]

    return sorted_rows

def organize_contours_top(contours, filename, max_rows=None):
    """
    Organizes contours into `max_rows` using **top edge clustering**, ensuring:
    - The row median is calculated using only the **50% closest** boxes.
    - Misaligned boxes are reassigned to the best row.
    - No row has more than `max_columns` boxes.

    Parameters:
    --------------
    contours: list
        List of bounding boxes (contours) detected in the image.

    max_rows: int
        Expected number of rows.

    max_columns: int
        Expected number of columns.

    Returns:
    --------------
    sorted_rows: list of lists
        Bounding boxes grouped into ordered rows.
    """

    if not contours:
        return []

    # Step 1: Extract **top edge y-coordinates** for clustering
    top_edges = np.array([cv2.boundingRect(c)[1] for c in contours]).reshape(-1, 1)
    # print(top_edges)
    # Step 2: Perform K-Means clustering on top edges
    # if max_rows is None:
    #     max_rows = get_max_rows_from_filename(filename)
    kmeans = KMeans(n_clusters=min(max_rows, len(top_edges)), random_state=0, n_init=50, tol=1e-2)
    kmeans.fit(top_edges)
    labels = kmeans.labels_

    # Step 3: Assign contours to rows based on clustering
    row_dict = {i: [] for i in range(max_rows)}
    for label, contour in zip(labels, contours):
        row_dict[label].append(contour)

    # Step 4: Compute **super strict row medians** using only the closest 50%
    row_medians = {}

    for i, row in row_dict.items():
        if len(row) > 2:
            y_tops = np.array([cv2.boundingRect(c)[1] for c in row])
            y_tops.sort()

            # Keep **only the middle 50% closest values**
            trimmed_y_tops = y_tops[len(y_tops) // 4 : 3 * len(y_tops) // 4]

            # Compute median of the **trimmed** values
            row_medians[i] = np.median(trimmed_y_tops)
        else:
            row_medians[i] = np.median([cv2.boundingRect(c)[1] for c in row])

    # Step 5: Move misplaced boxes to the best row
    adjusted_rows = {i: [] for i in range(max_rows)}

    for i, row in row_dict.items():
        for box in row:
            y_top = cv2.boundingRect(box)[1]  # **Top edge y-coordinate**
            
            # Find closest **trusted** row median (ignoring outliers)
            closest_row = i
            min_distance = abs(y_top - row_medians[i])

            if i > 0:  # Check row above
                distance_up = abs(y_top - row_medians[i-1])
                if distance_up < min_distance:
                    min_distance = distance_up
                    closest_row = i-1

            if i < max_rows - 1:  # Check row below
                distance_down = abs(y_top - row_medians[i+1])
                if distance_down < min_distance:
                    closest_row = i+1

            adjusted_rows[closest_row].append(box)

    # Step 6: Sort each adjusted row again (left to right)
    for i in range(len(adjusted_rows)):
        adjusted_rows[i] = sorted(adjusted_rows[i], key=lambda c: cv2.boundingRect(c)[0])

    # Convert to final sorted list
    sorted_rows = [adjusted_rows[i] for i in range(max_rows)]

    return sorted_rows




def transcription(
        detected_table_cells, ocr_model, tesseract_path, transient_transcription_output_dir, 
        pre_QA_QC_transcribed_hydroclimate_data_dir_station, station, month_filename, 
        no_of_rows, no_of_columns, no_of_rows_including_headers, max_columns, max_rows,
        max_cell_width_threshold, max_cell_height_threshold, train=False):
    
    log_mem("start transcription")

    if not train:
        if ocr_model == 'Tesseract-OCR':
            ## Lauching Tesseract-OCR
            pytesseract.pytesseract.tesseract_cmd = tesseract_path ## Here input the PATH to the Tesseract executable on your computer. See more information here: https://pypi.org/project/pytesseract/
        # if ocr_model == 'PaddleOCR':
        #     ## Lauching PaddleOCR, which would be used by downloading necessary files as shown below
        #     paddle_ocr = PaddleOCR(use_angle_cls=True, lang = 'en', use_gpu=False) ## Run only once to download all required files
        # if ocr_model == 'EasyOCR':
        #     ## Lauching EasyOCR
        # easyocr_reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

    # Contours of bounding boxes detected in cell recognition
    new_contours = detected_table_cells[0]

    image_with_all_bounding_boxes = detected_table_cells[1]
    table_copy = detected_table_cells[2] # Binarized table image using Adaptive Thresholding
    # table_copy = detected_table_cells[3] # Table image but as a clip of the original image (No Binarization)

    # Get the dimensions of the loaded image. Here, particulary the image/table width is very important for the column placement of cells/bounding boxes
    image_height, image_width, image_channels = image_with_all_bounding_boxes.shape

    # Image onto which the regions of interest (ROIs) i.e the cells, will be drawn for illustration purposes
    ROIs_image = table_copy.copy()

    results = []

    
    # Create ZIP archive for saving sorted images
    sorted_images_zip_path = os.path.join(transient_transcription_output_dir, station, "sorted_detected_cells_images.zip")
    sorted_images_zip = zipfile.ZipFile(sorted_images_zip_path, 'a', compression=zipfile.ZIP_DEFLATED)
    sorted_image_names_written = set()

    # Here we use two methods to arrange the boundung boxes (as a double check): (1) Using the middle coordinates of the bounding boxes , and (2) Using the top coordinated of the boudning boxes.
    organize_methods = {  
        # 'Midpoint': organize_contours_midpoint,
        'Top': organize_contours_top
    }

    # organize_methods = {  
    #         'Midpoint': organize_contours_by_column,
    #         'Top': organize_contours_by_column
    #     }

    for method_name, organize_method in organize_methods.items():

        ## Create an Excel workbook and add a worksheet where the transcribed text will be saved
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = 'OCR_Results'

        # Organize the contours
        # organized_rows = organize_method(new_contours, no_of_rows)

        sorted_rows = detected_table_cells[0]
        # Get the expected max rows (dynamically)
        # max_rows = get_max_rows_from_filename(month_filename)

        # Calculate how many rows need to be padded
        # missing_rows = 43 - max_rows  # Always ensuring 43 rows
        # missing_rows = max_rows - len(sorted_rows)  # Calculate how many rows are missing to reach the maximum number of rows

        # # If there are missing rows, pad `None` before the last two rows (totals & averages)
        # if missing_rows > 0:
        #     # Ensure we don't disturb the last two rows
        #     sorted_rows = sorted_rows[:-2] + [[None]] * missing_rows + sorted_rows[-2:]

        # sorted_rows = add_missing_boxes(sorted_rows, max_cell_width_threshold*0.7, max_cell_height_threshold*0.4)  # Add missing boxes per row where there are gaps

        ## FOR ILLUSTRATION PURPOSES: Uncomment the following lines if you want to see an example of how the sorting function works. This is for illustration purposes only and not necessary for the main functionality of the script. 
        
        # Create a copy of the image for visualization
        # image_before_sorting = table_copy.copy()
        image_after_sorting = table_copy.copy()

        # Generate colors for 43 rows. For sorted row visualitation purposes
        colors = generate_random_colors(max_rows)  # Random colours for the maximum number of rows, such that each row has its own color for easy identification

        # Draw markers on the image before sorting
        # draw_row_markers_and_boxes(image_before_sorting, organized_rows, colors)  # Green color for original order

        # Draw markers on the image after sorting
        draw_row_markers_and_boxes(image_after_sorting, sorted_rows, colors)  # Red color for sorted order

        # Ensure save directory exists
        save_dir = os.path.join(transient_transcription_output_dir, station)
        os.makedirs(save_dir, exist_ok=True)

        # # # Save or display the images for inspection
        print(method_name)
        # plt.imshow(image_before_sorting)
        # plt.show()
        # cv2.namedWindow(f'{method_name} before sorting', cv2.WINDOW_KEEPRATIO)
        # cv2.imshow(f'{method_name} before sorting', image_before_sorting)
        # cv2.resizeWindow(f'{method_name} before sorting', 1080, 720)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        # # plt.imshow(image_after_sorting)
        # # plt.show()
        # cv2.namedWindow(f'{method_name} after sorting', cv2.WINDOW_KEEPRATIO)
        # cv2.imshow(f'{method_name} after sorting', image_after_sorting)
        # cv2.resizeWindow(f'{method_name} after sorting', 1080, 720)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    
        # Create filename for after-sorting image
        zip_filename = f"after_sorting_{method_name}_{month_filename}.png"

        # Save the sorted image in the zipped file and overwrite if it already exists while suppressing the warning
        _, encoded = cv2.imencode('.png', image_after_sorting)
        sorted_images_zip.writestr(zip_filename, encoded.tobytes())
        sorted_image_names_written.add(zip_filename)

        # Ensure save directory exists
        save_dir = os.path.join(transient_transcription_output_dir, station)
        os.makedirs(save_dir, exist_ok=True)

        zip_filename = os.path.splitext(os.path.split(month_filename)[-1])[0]
        # Open zip file to save all the clipped cells images
        zip_save_path = os.path.join(transient_transcription_output_dir, station, f"{zip_filename}_clipped_cells.zip")
        roi_zip = zipfile.ZipFile(zip_save_path, 'w', compression=zipfile.ZIP_DEFLATED)

        written_filenames = set()

        log_mem("starting OCR loop")
        # def create_cell_hash_tables(organized_cols, organized_rows):
        #     """
        #     Create hash tables for O(1) cell lookups
        #     """
        #     # Hash table: bounding_box -> column_index
        #     bbox_to_col = {}
        #     for col_index, col in enumerate(organized_cols, start=1):
        #         for contour in col:
        #             bbox = tuple(cv2.boundingRect(contour))
        #             bbox_to_col[bbox] = col_index
            
        #     # Hash table: bounding_box -> row_index  
        #     bbox_to_row = {}
        #     for row_index, row in enumerate(organized_rows, start=1):
        #         if row is None:
        #             continue
        #         for contour in row:
        #             bbox = tuple(cv2.boundingRect(contour))
        #             bbox_to_row[bbox] = row_index
            
        #     return bbox_to_col, bbox_to_row

        # # IMPORTANT: Use sorted_rows for BOTH hash table creation AND lookup
        # bbox_to_col, bbox_to_row = create_cell_hash_tables(organized_cols, sorted_rows)  # Use sorted_rows here
        # cell_references = {}

        # # Now O(1) lookup using the same sorted_rows
        # for row_index, row in tqdm(enumerate(sorted_rows, start=1)):
        #     if row is None:
        #         continue
                
        #     for contour in row:
        #         bbox = tuple(cv2.boundingRect(contour))
                
        #         # O(1) lookup
        #         col_index = bbox_to_col.get(bbox)
        #         row_index_from_hash = bbox_to_row.get(bbox)  # Rename to avoid conflict
                
        #         if col_index is None or row_index_from_hash is None:
        #             print(f'Could not determine cell reference for contour at {bbox}')
        #             continue
                
        #         x_0, y_0 = bbox[:2]
        #         cell_references[(x_0, y_0)] = (row_index_from_hash, col_index)
        # print('cell_references:', cell_references)
        # assigned_columns_per_row = {}
        for row_index, row in tqdm(enumerate(sorted_rows, start=1), total=len(sorted_rows), desc="Processing rows"):
            
            # log_mem(f"processing row {row_index}")
            if row is None:
                continue  # Skip processing for empty rows
            row = sorted(row, key=lambda c: cv2.boundingRect(c)[0])  # Sort cells left to right within the row
            for col_index, contour in enumerate(row, start=1):

                x_0, y_0, w, h = cv2.boundingRect(contour)

                # Define increase factors for bounding box modification
                increase_factor_width = 0.07  # Increase width by 7%
                increase_factor_height = 0.20  # Increase height by 20%

                # Expand width while keeping it centered
                new_w = int(w * (1 + increase_factor_width))  # Increase width
                x = max(0, x_0 - (new_w - w) // 2)  # Adjust x to keep center fixed

                # Expand height symmetrically
                new_h = int(h * (1 + increase_factor_height * 2))  # Increase height
                y = max(0, y_0 - (new_h - h) // 2)  # Adjust y to keep center fixed

                # Ensure bounding box remains within valid image bounds
                x = max(0, x)
                y = max(0, y)
                w = max(1, new_w)  # Avoid zero or negative width
                h = max(1, new_h)  # Avoid zero or negative height

                 # Draw bounding box on the visualization image
                cv2.rectangle(ROIs_image, (x, y), (x + w, y + h), (0, 0, 255), 3)

                # ********
                # OCR
                # Crop each cell using the bounding rectangle coordinates
                ROI = table_copy[y:y+h, x:x+w]  # Ensure consistency with visualization bounding boxes
                # ********

                # ### FOR ILLUSTRATION PURPOSES: This line below is about drawing a rectangle on the image with the shape of the bounding box. Its not needed for the OCR. This is only for debugging purposes.
                # # image_with_all_bounding_boxes = cv2.rectangle(image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 5)

                # # # Draw the adjusted ROI on the output image
                # # cv2.rectangle(ROIs_image, (x, y), (x + w, y + h), (0, 255, 0), 5)  # (0, 255, 0) represent a green color for ROI, and 5 is the thicnkess of the ROI bounbdary boxes
                
                # # Draw the updated bounding box
                # cv2.rectangle(ROIs_image, (x, y), (x + new_w, y + new_h), (0, 255, 0), 4)

                
                # OCR
                
                if ROI.size != 0:  # Check if the height and width are greater than zero. This is to prevent invalid ROIs
                    
                    # Save the detected text image/ROI
                    save_dir =  os.path.join(transient_transcription_output_dir, station)
                    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
                    save_path_detected_text = os.path.join(save_dir, 'detected.png')
                    cv2.imwrite(save_path_detected_text, ROI)

                    cell_ref = f'{get_column_letter(col_index)}{row_index}'
                    # write cell ref on full table image for reference
                    # cv2.putText(image_after_sorting, adjusted_cell_ref, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        

                    if not train:
                        if ocr_model == 'Tesseract-OCR':
                        # Using Tesseract-OCR
                            ocr_result = pytesseract.image_to_string(save_path_detected_text, lang='cobecore_finetuned', config='--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789') # Just added -c tessedit_char_whitelist=0123456789 to really limit the text type/values detected

                            # Here's a brief explanation of some Page Segmentation Modes (PSMs) available in Tesseract:
                            # 0: Orientation and script detection (OSD) only.
                            # 1: Automatic page segmentation with OSD.
                            # 2: Automatic page segmentation, but no OSD, or OCR.
                            # 3: Fully automatic page segmentation, but no OSD. (Default)
                            # 4: Assume a single column of text of variable sizes.
                            # 5: Assume a single uniform block of vertically aligned text.
                            # 6: Assume a single uniform block of text.
                            # 7: Treat the image as a single text line.
                            # 8: Treat the image as a single word.
                            # 9: Treat the image as a single word in a circle.
                            # 10: Treat the image as a single character.
                            # 11: Sparse text. Find as much text as possible in no particular order.
                            # 12: Sparse text with OSD.
                            # 13: Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

                        # Uncomment the following lines if you'd like to make use of PaddleOCR
                        # if ocr_model == 'PaddleOCR':
                        #     ## Using PaddleOCR
                        #     ocr_result = paddle_ocr.ocr('detected.png', cls = True)

                        # if ocr_model == 'EasyOCR':
                        # # Using EasyOCR
                        #     ocr_result = easyocr_reader.readtext(save_path_detected_text, detail = 0, allowlist='0123456789')
                        #     # In EasyOCR, the detail parameter specifies the level of detail in the output. 
                        #             # When using the readtext method, the detail parameter can be set to different values to control what kind of output you get. 
                        #             # Specifically:
                        #             # detail=1: The output will be a list of tuples, where each tuple contains detailed information about the detected text, 
                        #             # including the bounding box coordinates, the text string, and the confidence score. Example: [(bbox, text, confidence), ...].

                        #             # detail=0: The output will be a list of strings, where each string is the detected text without any additional details. 
                        #             # This is a simpler output format that only provides the recognized text. Example: ["text1", "text2", ...].
                        #     if isinstance(ocr_result, list): # This is because EasyOCR's results are returned as a list
                        #             ocr_result = ''.join(ocr_result)  # Convert list to a string
                        
                        
                        # Using OCR for handwritten text recognition
                        # if ocr_result is not None:
                            
                            # if not ocr_result.strip(): # Check if the result is empty or only whitespace. This could be due to the selected OCR (in this case: Tesseract-OCR) not being able to recognize the text in the ROI.
                            #     # For this reason, we can try another OCR, say for example Easy OCR, to try to recognize the text in this ROI
                            #     ocr_result = easyocr_reader.readtext(save_path_detected_text, detail = 0, allowlist='0123456789')
                            #     if isinstance(ocr_result, list): # This is because EasyOCR's results are returned as a list
                            #         ocr_result = ''.join(ocr_result)  # Convert list to a string



                                     
                        # Attain the Ms Excel Template cell coordinates
                        # # Determine the cell reference using the x coodrinate of the bounding box, row index, maximum column number, and image/table width
                        # Transform col index to Excel-style letter (A, B, C, ..., Z, AA, AB, etc.)
                        # cell_ref = calculate_cell_reference(x, w, row_index, assigned_columns_per_row, max_columns=24, table_width=image_width) # e.g., A1, B5, etc.
                        
                        # Place the OCR/HTR recognized text in its respective Ms Excel cell 
                        ws[cell_ref].value = ocr_result.strip()  # Remove leading/trailing whitespace   

                        # Set up border styles for excel output
                        thin_border = Border(
                            left=Side(style='thin'),
                            right=Side(style='thin'),
                            top=Side(style='thin'),
                            bottom=Side(style='thin'))

                        # Loop through cells to apply borders
                        for row in ws.iter_rows(min_row=1, max_row=no_of_rows, min_col=1, max_col=no_of_columns):
                            for cell in row:
                                cell.border = thin_border

                        
                        ## CONTINOUS LEARNING (MACHINE-LEARNING) OF THE OCR. # Save clipped cells as images for a training dataset
                        # print(cell_ref)
                        # Adjust cell_ref to match final Excel layout: 2 columns inserted + 3 header rows added
                        # adjusted_column_index = openpyxl.utils.column_index_from_string(cell_ref[0]) #+ 2  # +2 for two inserted columns on the left side of the excel sheet. In our case these were the 'Pentad No.' and the Date. Change this depending on your sheets or make it +0 (Zero) if there were no extra columns to the left
                        # adjusted_row_index = int(cell_ref[1:]) #+ int(no_of_rows_including_headers - no_of_rows)  # In our case: = +3 for three inserted header rows. See # SPECIAL ADDITION TO CODE / CUSTOMIZATION below

                        # # Convert adjusted column index back to letter
                        # adjusted_column_letter = openpyxl.utils.get_column_letter(adjusted_column_index)

                        # Construct adjusted Excel-style reference
                        # adjusted_cell_ref = f"{adjusted_column_letter}{adjusted_row_index}"
                        adjusted_cell_ref = cell_ref  # Here we assume no adjustments are needed. Change this if your Excel layout has extra rows/columns

                        # write cell ref on full table image for reference
                        cv2.putText(image_after_sorting, adjusted_cell_ref, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        if method_name == 'Top': # Since top cordinates of the bounding boxes are the most prefered organizing method here
                            # Save the clipped images inside the loop:
                            roi_filename = f"{month_filename}_{adjusted_cell_ref}.png"

                            if roi_filename in written_filenames:
                                continue
                            
                            written_filenames.add(roi_filename) # Mark it as saved

                            _, encoded = cv2.imencode('.png', ROI)
                            roi_zip.writestr(roi_filename, encoded.tobytes())

                    else:
                        print('No values detected in clip')
                else:
                    print('ROI is empty or invalid')
        full_table_save_dir = os.path.join(save_dir, f'full_table_{month_filename}.jpg')
        os.makedirs(os.path.dirname(full_table_save_dir), exist_ok=True)
        print(full_table_save_dir)
        cv2.imwrite(full_table_save_dir, image_after_sorting)

    #     ### SPECIAL ADDITION TO CODE / CUSTOMIZATION: The lines below are particular to our tables. Here we replace the headers and row labels as in the original table format. This will vary from your setup and this should be customized to your particular sheets.
    #     # Insert two columns on the left side of the excel sheet
    #     ws.insert_cols(1, 2)
        
    #     # Insert a new row at the top for headers
    #     ws.insert_rows(1, amount = 3)

    #     # Define your headers (adjust as needed)
    #     headers =       ["", "Date", "Luchtdruk", "",   "",   "Temperatuur", "",    "",      "",    "",      "",    "",          "",        "",        "Dampdrukking", "",   "",   "Vochtigheid", "",   "",   "Wind", "",   "",   "",   "",   "",  "Bewolking", "",   "",   "",    "Neerslag",   "",         "",          "",]
    #     sub_headers_1 = ["", "",     "8a",        "2p", "6p", "8a",          "",    "2p",    "",    "6p",    "",    "Som droog", "Maximum", "Minimum", "8a",           "2p", "6p", "8a",          "2p", "6p", "8a",   "",   "2p", "",   "6p", "",  "8a",        "2p", "6p", "Som", "Oranjestad", "Bengalen", "Zeelandia", "Farm"]
    #     sub_headers_2 = ["", "",     "",          "",   "",   "Droog",       "Nat", "Droog", "Nat", "Droog", "Nat", "",          "",        "",        "",             "",   "",   "",            "",   "",   "Ri",   "Kr", "Ri", "Kr", "Ri", "Kr" "",          "",   "",   "",    "",           "",         "",          "",]

    #     # Add the headers to the first row
    #     for col_num, header in enumerate(headers, start=1):
    #         cell = ws.cell(row=1, column=col_num, value=header)
    #         if header == "No de la pentade" or header == "Date" or header == "Bellani (gr. Cal/cm2) 6-6h" or header == "Pluies en mm. 6-6h":
    #             cell.alignment = Alignment(textRotation=90)

    #     # Add the first row of sub-headers to the second row
    #     for col_num, sub_header in enumerate(sub_headers_1, start=1):
    #         ws.cell(row=2, column=col_num, value=sub_header)

    #     # Add the second row of sub-headers to the third row
    #     for col_num, sub_header in enumerate(sub_headers_2, start=1):
    #         ws.cell(row=3, column=col_num, value=sub_header)
        
    #     # Merge cells for multi-column headers
    #     ws.merge_cells(start_row=1, start_column=1, end_row=3, end_column=1) #No de la pentade
    #     ws.merge_cells(start_row=1, start_column=2, end_row=3, end_column=2) #Date
    #     ws.merge_cells(start_row=1, start_column=3, end_row=3, end_column=3) #Bellani
    #     ws.merge_cells(start_row=1, start_column=4, end_row=1, end_column=8) #Températures extrêmes
    #     ws.merge_cells(start_row=1, start_column=9, end_row=1, end_column=10) #Evaportation
    #     ws.merge_cells(start_row=1, start_column=11, end_row=3, end_column=11) #Pluies
    #     ws.merge_cells(start_row=1, start_column=12, end_row=1, end_column=16) #Température et Humidité de l'air à 6 heures
    #     ws.merge_cells(start_row=1, start_column=17, end_row=1, end_column=21) #Température et Humidité de l'air à 15 heures
    #     ws.merge_cells(start_row=1, start_column=22, end_row=1, end_column=26) #Température et Humidité de l'air à 18 heures
    #     ws.merge_cells(start_row=1, start_column=27, end_row=3, end_column=27) #Date
    #     # subheaders
    #     ws.merge_cells(start_row=2, start_column=4, end_row=2, end_column=7) #Abri
    #     ws.merge_cells(start_row=2, start_column=9, end_row=2, end_column=10) #Piche
    #     ws.merge_cells(start_row=2, start_column=12, end_row=2, end_column=16) #(Psychromètre a aspiration)
    #     ws.merge_cells(start_row=2, start_column=17, end_row=2, end_column=21) #(Psychromètre a aspiration)
    #     ws.merge_cells(start_row=2, start_column=22, end_row=2, end_column=26) #(Psychromètre a aspiration)


    #     # Label Date, Total and Average rows
    #     row_labels = ["1","2", "3", "4", "5", "6", "7", "8", "9", "10", "Som.", 
    #                   "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "Som.", 
    #                   "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", 
    #                   "Som.", "Gem.", "Som tot.", "Gem tot."]
    #     # Update the cells in the second and last column with the date values
    #     columns = [2, 27]
    #     for col in columns:
    #         for i, value in enumerate(row_labels, start=4):
    #             cell = ws.cell(row=i, column=col)
    #             cell.value = value
    #             # wb.save(new_version_of_file) # Save the modified workbook
        
    #     # Save Excel file
        file_path = os.path.join(save_dir, month_filename, f'{method_name}_Excel_with_OCR_Results.xlsx')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        wb.save(file_path)
        results.append([file_path])

        roi_zip.close()  
        wb.close()
        sorted_images_zip.close()
        log_mem("finished OCR loop")

        
    #     ### FOR ILLUSTRATION PURPOSES: Uncomment the following lines if you want to see the ROIs per table
    #     # plt.imshow(ROIs_image)
    #     # plt.show()
    
    # # Update paths to include the new Excel files (from the methods above) for merging. This is just a double check for correct cell placement
    # top_excel_path = os.path.join(save_dir, 'Top_Excel_with_OCR_Results.xlsx')
    # midpoint_excel_path = os.path.join(save_dir, 'Midpoint_Excel_with_OCR_Results.xlsx')
    # path_to_save_merged_excel_file = os.path.join(pre_QA_QC_transcribed_hydroclimate_data_dir_station, f'{month_filename}_pre_QA_QC.xlsx')
    # merge_excel_files(top_excel_path, midpoint_excel_path, path_to_save_merged_excel_file, 1, no_of_rows_including_headers) # this prioritizes the top coordinates of the bounding box to the mid point coordinates when placing the transcribed data into an excel sheet. But considers the best placement for both as double check. Here the 1 represent the first row and the 46 represents the last possible row to perform the merging of the excel files. In our case we have 43 rows with data + 3 header rows = 46.
    # # merge_excel_files(midpoint_excel_path, top_excel_path, path_to_save_merged_excel_file, 1, no_of_rows_including_headers) # this prioritizes the midpoint coordinates of the bounding box to the top coordinates when placing the transcribed data into an excel sheet. But considers the best placement for both as double check. Here the 1 represent the first row and the 46 represents the last possible row to perform the merging of the excel files. In our case we have 43 rows with data + 3 header rows = 46.

    return zip_save_path