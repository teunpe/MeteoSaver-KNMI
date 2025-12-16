import os
import cv2
import numpy as np
from src.table_and_cell_detection_model import remove_vertical_lines, filter_contours
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import logging
from sklearn.cluster import KMeans

from src.transcription import organize_contours_midpoint, organize_contours_top, get_max_rows_from_filename#, add_missing_boxes, 
from src.transcription import generate_random_colors, draw_row_markers_and_boxes, calculate_trimmed_mean
from openpyxl import Workbook

logger = logging.getLogger('meteosaver')


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

def table_and_cell_detection(
        image_in_grayscale, binarized_image, original_image, station, 
        month_filename, transient_transcription_output_dir, clip_up, 
        clip_down, clip_left, clip_right, max_table_width, max_table_height, 
        min_cell_width_threshold, min_cell_height_threshold, 
        max_cell_width_threshold, max_cell_height_threshold, 
        space_height_threshold, space_width_threshold, max_cell_height_per_box, 
        no_of_rows, no_of_columns) -> list:
    
    table_original_image = original_image
    full_detected_table_with_labels = binarized_image

    table_img_bin = remove_vertical_lines(binarized_image)
    
    # Save the binary image for use later in detecting text
    save_dir = os.path.join(transient_transcription_output_dir, station)
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    save_path = os.path.join(save_dir, 'table_binarized.jpg')
    cv2.imwrite(save_path, table_img_bin)

    # Invert the binarized image of the table
    img_bin = 255-table_img_bin
    # Detect the vertical lines in the image
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(table_img_bin).shape[1]//50)) # The '//50' divides the length of the array (table) by 50, likely to obtain a fraction of the length for the structuring element,
    eroded_image = cv2.erode(img_bin, vertical_kernel, iterations=1)
    vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=5)
    # Detect the horizontal lines in the image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(table_img_bin).shape[1]//20, 1)) # The '//20' divides the width of the array (table) by 20, likely to obtain a fraction of the width for the structuring element.
    eroded_image= cv2.erode(img_bin, hor_kernel, iterations=1)
    horizontal_lines = cv2.dilate(eroded_image, hor_kernel, iterations=5)
    # Blending the imaegs with the vertical lines and the horizontal lines 
    combined_vertical_and_horizontal_lines = cv2.addWeighted(vertical_lines, 1, horizontal_lines, 1, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    combined_image_dilated = cv2.dilate(combined_vertical_and_horizontal_lines, kernel, iterations=5)
    # Remove the lines from the image (table)
    image_without_lines = cv2.subtract(img_bin, combined_image_dilated)
    # Remove smaller 'still-visible' lines through noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image_without_lines_noise_removed = cv2.erode(image_without_lines, kernel, iterations=1)
    image_without_lines_noise_removed = cv2.dilate(image_without_lines_noise_removed, kernel, iterations=1)


    # plt.imshow(combined_image_dilated, cmap="gray")
    # plt.title("First Filtered Image - No Dots - No Lines")
    # plt.show()

    # Step 1: Detect all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_without_lines_noise_removed, connectivity=8)

    # Create a mask to keep only valid text components
    filtered_image = np.zeros_like(image_without_lines_noise_removed)

    # Set thresholds
    ASPECT_RATIO_THRESHOLD = 4  # If Width / Height > 4 (was 4 before this new check), it's considered horizontal noise
    HEIGHT_THRESHOLD = 5  # Remove any blobs with height less than this
    PROXIMITY_THRESHOLD = 5  # Maximum distance (in pixels) to consider a dot "close" to a number

    # Store y-coordinates and x-limits of horizontal dots **ONLY NEAR TEXT**
    horizontal_lines = []

    logger.debug('Step 2: Identify large text components (potential numbers)')
    text_components = []
    dots_to_remove = np.zeros_like(labels)  # Mask for dots to be removed

    for i in tqdm(range(1, num_labels)):
        x, y, w, h, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]

        aspect_ratio = w / h  # Compute aspect ratio

        # Keep only real text (numbers) as reference components
        if area > 100 and aspect_ratio < ASPECT_RATIO_THRESHOLD and h > HEIGHT_THRESHOLD:   # was 100 before this check, then i tried 50
            text_components.append((x, y, w, h))
            filtered_image[labels == i] = 255  # Keep text
        
        else:
            dots_to_remove[labels == i] = 255  # Mark dots for removal

    # plt.imshow(filtered_image, cmap="gray")
    # plt.title("First Filtered Image - No Dots - No Lines")
    # plt.show()

    # Step 1: Detect all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_without_lines_noise_removed, connectivity=8)

    # Create a mask to keep only valid text components
    filtered_image = np.zeros_like(image_without_lines_noise_removed)

    # Set thresholds
    ASPECT_RATIO_THRESHOLD = 4
    HEIGHT_THRESHOLD = 5
    PROXIMITY_THRESHOLD = 5

    horizontal_lines = []

    logger.debug('Step 2: Identify large text components - OPTIMIZED')

    # Pre-compute aspect ratios for all components
    if num_labels > 1:
        widths = stats[1:, cv2.CC_STAT_WIDTH]
        heights = stats[1:, cv2.CC_STAT_HEIGHT]
        areas = stats[1:, cv2.CC_STAT_AREA]
        aspect_ratios = widths / heights
        
        text_components = []
        text_labels_to_keep = []
        dot_labels_to_remove = []
        
        # Single loop with pre-computed values
        for i in range(num_labels - 1):  # Exclude background
            label_id = i + 1
            area = areas[i]
            aspect_ratio = aspect_ratios[i]
            height = heights[i]
            
            # Keep only real text (numbers) as reference components
            if area > 100 and aspect_ratio < ASPECT_RATIO_THRESHOLD and height > HEIGHT_THRESHOLD:
                x, y, w, h = stats[label_id, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
                text_components.append((x, y, w, h))
                text_labels_to_keep.append(label_id)
            else:
                dot_labels_to_remove.append(label_id)
        
        # Batch update images
        if text_labels_to_keep:
            text_mask = np.isin(labels, text_labels_to_keep)
            filtered_image[text_mask] = 255
        
        dots_to_remove = np.zeros_like(labels)
        if dot_labels_to_remove:
            dot_mask = np.isin(labels, dot_labels_to_remove)
            dots_to_remove[dot_mask] = 255
    else:
        text_components = []
        dots_to_remove = np.zeros_like(labels)

    # plt.imshow(filtered_image, cmap="gray")
    # plt.title("First Filtered Image - No Dots - No Lines")
    # plt.show()

    # Pre-convert text_components to numpy arrays for faster operations
    if text_components:
        text_array = np.array(text_components)  # Shape: (N, 4) for N text components
        
        for i in tqdm(range(1, num_labels)):
            x, y, w, h, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
            aspect_ratio = w / h
            # Skip already kept text components
            if area > 100 and aspect_ratio < ASPECT_RATIO_THRESHOLD and h > HEIGHT_THRESHOLD:
                continue
            
            # Vectorized proximity check
            dot_center_x, dot_center_y = x + w // 2, y + h // 2
            
            # Check proximity to all text components at once
            in_x_range = ((text_array[:, 0] - PROXIMITY_THRESHOLD <= dot_center_x) & 
                        (dot_center_x <= text_array[:, 0] + text_array[:, 2] + PROXIMITY_THRESHOLD))
            in_y_range = ((text_array[:, 1] - PROXIMITY_THRESHOLD <= dot_center_y) & 
                        (dot_center_y <= text_array[:, 1] + text_array[:, 3] + PROXIMITY_THRESHOLD))
            
            keep_dot = np.any(in_x_range & in_y_range)
            
            if keep_dot:
                filtered_image[labels == i] = 255
                horizontal_lines.append((dot_center_y, x, x + w))

    plot = False
    figsize = (10,10)

    # config.read('configuration.ini')
    # min_cell_width_threshold = int(config['TableAndCellDetection']['min_cell_width_threshold']) # Minimum cell width threshold for table and cell detection
    # max_cell_width_threshold = int(config['TableAndCellDetection']['max_cell_width_threshold']) # Maximum cell width threshold for table and cell detection
    # min_cell_height_threshold = int(config['TableAndCellDetection']['min_cell_height_threshold']) # Minimum cell height threshold for table and cell detection
    # max_cell_height_threshold = int(config['TableAndCellDetection']['max_cell_height_threshold']) # Maximum cell height threshold for

    # max_cell_width_threshold = 300
    # max_cell_height_threshold = 65


    # Step 2: Use **horizontal dilation** to merge digits within numbers, but prevent full merging
    kernel_to_remove_gaps = np.ones((1, 6), np.uint8)  # Horizontal merging kernel
    image_with_number_blobs = cv2.dilate(filtered_image, kernel_to_remove_gaps, iterations=5)  # Controlled dilation



    # Step 3: Apply **morphological closing** to ensure numbers remain compact blobs
    rect_kernel = np.ones((1, 3), np.uint8)   # was 3 initially. in prev step was 1
    image_with_word_blobs = cv2.morphologyEx(image_with_number_blobs, cv2.MORPH_CLOSE, rect_kernel, iterations=3)


    # Define a horizontal erosion kernel
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))  # (height, width) to break horizontal merges. had it at 1,5 initially 
    # Apply erosion to break the connection between digits
    image_with_word_blobs = cv2.erode(image_with_word_blobs, horizontal_kernel, iterations=2)  # Increase iterations if still connected. Had 2 iteratons initially

    if plot:
        plt.figure(figsize=figsize)
        plt.imshow(image_with_word_blobs, cmap = 'gray') # figure showing detected table image with horizintal and vertical lines removed.
        plt.title('after erosion')
        plt.show()

    # ** delete small horizontal lines.
    # Create a blank mask for horizontal lines
    # horizontal_line_mask = np.zeros_like(image_with_word_blobs)


    # Draw **thin** horizontal lines at detected y-positions (but not subtracting yet)
    # for y, x_start, x_end in horizontal_lines:
    #     cv2.line(horizontal_line_mask, (x_start - 15, y), (x_end + 15, y), 255, thickness=3)  # Initially very thin

    # **Step 3: Subtract the dilated lines from the filtered image**
    # image_with_word_blobs = cv2.subtract(image_with_word_blobs, horizontal_line_mask)




    # Subtract the detected vertical and horizontal lines from the filtered image**
    image_with_word_blobs = cv2.subtract(image_with_word_blobs, combined_image_dilated)

    if plot:
        plt.figure(figsize=figsize)
        plt.imshow(image_with_word_blobs, cmap = 'gray') # figure showing detected table image with horizintal and vertical lines removed.
        plt.title('subtraction of detected vertical and horizontal lines of table')
        plt.show() 


    # # Define a horizontal erosion kernel
    # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))  # (height, width) to break horizontal merges. had it at 1,5 initially 
    # # Apply erosion to break the connection between digits

    # # ** delete small horizontal lines.
    # # Create a blank mask for horizontal lines
    # horizontal_line_mask = np.zeros_like(image_with_word_blobs)

    # # Draw **thin** horizontal lines at detected y-positions (but not subtracting yet)
    # for y, x_start, x_end in horizontal_lines:
    #     cv2.line(horizontal_line_mask, (x_start - 15, y), (x_end + 15, y), 255, thickness=3)  # Initially very thin


    # **Step 2: Find Wide Blobs That Need Stronger Erosion**
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image_with_word_blobs, connectivity=8)

    for i in range(1, num_labels):  # Ignore background (label 0)
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        
        if h > max_cell_height_threshold: # ** If the blob is too tall, apply stronger erosion along the horizontal dimension**
            # erosion_kernel_vert = np.ones((min(h // 15, 12), 6), np.uint8)  # Adaptive vertical erosion # Initially had this: np.ones((min(h // 20, 12), 3), np.uint8)
            # erosion_kernel_vert = np.ones((1, 13), np.uint8)  # Slightly stronger erosion for wide blobs # Imitially had (10,3), BUT (1,10) improved it
            # roi = image_with_word_blobs[y:y+h, x:x+w]
            # eroded_roi = cv2.erode(roi, erosion_kernel_vert, iterations=3)  
            
            # # **Restore horizontal thickness**
            # dilation_kernel_vert = np.ones((1, 6), np.uint8)  # initially had (1,6)
            # recovered_roi = cv2.dilate(eroded_roi, dilation_kernel_vert, iterations=3)  

            # image_with_word_blobs[y:y+h, x:x+w] = recovered_roi 

            roi = image_with_word_blobs[y:y+h, x:x+w].copy()
            sliced_roi = roi.copy()

            # Determine how many rows this blob spans
            row_height = max_cell_height_threshold  # Approximate average row height in your dataset
            num_slices = h // row_height

            # Insert horizontal black lines to slice between rows
            for s in range(1, num_slices + 1):
                y_line = s * row_height
                if y_line < h:
                    cv2.line(sliced_roi, (0, y_line), (w, y_line), 0, thickness=3)
                    # print('Slided one blob horizontally') # Just a check to know how many wrongfully joined blobs in the same column and adjascent rows were sliced

            # Replace original ROI with sliced version
            image_with_word_blobs[y:y+h, x:x+w] = sliced_roi

    if plot:
        plt.figure(figsize=figsize)
        plt.imshow(image_with_word_blobs, cmap="gray")
        plt.title("After vertical erosion")
        plt.show()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image_with_word_blobs, connectivity=8)
    for i in range(1, num_labels):  # Ignore background (label 0)
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if w > max_cell_width_threshold:  # **If the blob is too wide, apply stronger erosion**
            erosion_kernel = np.ones((2, 2), np.uint8)  # Slightly stronger erosion for wide blobs
            roi = image_with_word_blobs[y:y+h, x:x+w]
            eroded_roi = cv2.erode(roi, erosion_kernel, iterations=3)  # More iterations for wider blobs'
            
            # **Apply vertical recovery dilation to restore eroded thickness**
            dilation_kernel = np.ones((2, 1), np.uint8)  # Small dilation to restore height
            recovered_roi = cv2.dilate(eroded_roi, dilation_kernel, iterations=3)  # Restore vertical thickness
            
            image_with_word_blobs[y:y+h, x:x+w] = recovered_roi  # Replace only this region


    if plot:
        plt.figure(figsize=figsize)
        plt.imshow(image_with_word_blobs, cmap="gray")
        plt.title("Fixing Over-Merging across rows and columns")
        plt.show()

    # # One more step with **horizontal dilation** to merge digits within numbers, but prevent full merging
    # kernel_to_remove_gaps = np.ones((1, 10), np.uint8)  # Horizontal merging kernel
    # image_with_number_blobs = cv2.dilate(filtered_image, kernel_to_remove_gaps, iterations=5)  # Controlled dilation
    # plt.imshow(image_with_word_blobs, cmap="gray")
    # plt.title("Last horizontal dilations to completey merge digits of numbers in one cell")
    # plt.show()



    # Remove the lines from the image (table)
    image_with_word_blobs = cv2.subtract(image_with_word_blobs, combined_image_dilated)

    if plot:
        plt.figure(figsize=figsize)
        plt.imshow(image_with_word_blobs, cmap="gray")
        plt.title("image_removing_previous_lines_areas_to_Avoid_merging_of_blobs_horizontally")
        plt.show()

    # # One more step to totally avoid joining cells in adjascent rows
    # # **Step 2: Find Wide Blobs That Need Stronger Erosion**
    # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image_with_word_blobs, connectivity=8)

    # for i in range(1, num_labels):  # Ignore background (label 0)
    #     x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
    #     if h > 70: # ** If the blob is too tall, apply stronger erosion along the horizontal dimension**
    #             # erosion_kernel_vert = np.ones((min(h // 15, 12), 6), np.uint8)  # Adaptive vertical erosion # Initially had this: np.ones((min(h // 20, 12), 3), np.uint8)
    #             erosion_kernel_vert = np.ones((1, 10), np.uint8)  # Slightly stronger erosion for wide blobs # Imitially had (10,3), BUT (1,10) improved it
    #             roi = image_with_word_blobs[y:y+h, x:x+w]
    #             eroded_roi = cv2.erode(roi, erosion_kernel_vert, iterations=3)  
                
    #             # # **Restore horizontal thickness**
    #             # dilation_kernel_vert = np.ones((1, 6), np.uint8)  # initially had (1,6)
    #             # recovered_roi = cv2.dilate(eroded_roi, dilation_kernel_vert, iterations=3)  

    #             image_with_word_blobs[y:y+h, x:x+w] = eroded_roi

    # plt.imshow(image_with_word_blobs, cmap="gray")
    # plt.title("h>70 blobs eroded even more ")
    # plt.show()

                

    ## Using contours in order to detect text in the table after removing the vertical and horizontal lines
    # Assuming 'table' is your input image in BGR format
    # result = cv2.findContours(image_without_lines_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.findContours(image_with_word_blobs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = result[0]
    # Original image of table in binarizesd format
    image_with_all_bounding_boxes = cv2.imread(save_path)
    table_binarized = image_with_all_bounding_boxes.copy()



    ## FOR VISUALIZATION PURPOSES. Uncomment the lines below to plot the identified cells (contours/bounding boxes)
    # Make a copy of the original image to overlay contours without modifying the original
    table_img_bin_overlayed_with_contours = table_img_bin.copy()
    # Convert the grayscale image to RGB to support colored bounding boxes
    table_img_bin_overlayed_with_contours = cv2.cvtColor(table_img_bin_overlayed_with_contours, cv2.COLOR_GRAY2RGB)

    from src.table_and_cell_detection_model import filter_contours

    # Filter out smaller or larger bounding boxes from all the detected text contours. This is helpful to avoid overly large cells or small cells with no text. Remember to adjust these values based on the table structure in your specific case 
    filtered_contours = filter_contours(contours, min_cell_width_threshold, min_cell_height_threshold, max_cell_width_threshold, max_cell_height_threshold)

    # Iterate over each contour in the new_contours list and draw bounding boxes
    for contour in filtered_contours:
        if contour is not None and len(contour) > 0:
            x, y, w, h = cv2.boundingRect(contour)

            # # Adjust bounding box dimensions
            # increase_factor_width = 0.05
            # increase_factor_height = 0.25
            # x += int(w * increase_factor_width) # Increase width
            # y -= int(h * increase_factor_height) # Increase height
            # w -= int(w * increase_factor_width) # Decrease width a little to avoid vertical lines that may be transcribed as the number 1 yet they aren't a number
            # h += int(h * increase_factor_height * 2) # Increase height

            # Define increase factors for bounding box modification
            increase_factor_width = 0.07  # Increase width by 7%
            increase_factor_height = 0.10  # Increase height by 20%

            # Expand width while keeping it centered
            new_w = int(w * (1 + increase_factor_width))  # Increase width
            x = max(0, x - (new_w - w) // 2)  # Adjust x to keep center fixed

            # Expand height symmetrically
            new_h = int(h * (1 + increase_factor_height * 2))  # Increase height
            y = max(0, y - (new_h - h) // 2)  # Adjust y to keep center fixed

            # Ensure bounding box remains within valid image bounds
            x = max(0, x)
            y = max(0, y)
            w = max(1, new_w)  # Avoid zero or negative width
            h = max(1, new_h)  # Avoid zero or negative height
            
            # Draw the bounding box directly on the overlay image
            cv2.rectangle(table_img_bin_overlayed_with_contours, (x, y), (x + w, y + h), (0, 0, 255), 3)

    if plot:
        # Display the image with bounding boxes using matplotlib
        plt.figure(figsize=figsize)
        plt.imshow(table_img_bin_overlayed_with_contours)
        plt.axis('off')  # Hide axis
        plt.show()

    # Sort contours by y-coordinate
    contours_sorted = sorted(filtered_contours, key=lambda c: cv2.boundingRect(c)[1])

    # Get the dimensions of the loaded image. Here, particulary the image/table width is very important for the column placement of cells/bounding boxes
    image_height, image_width, image_channels = image_with_all_bounding_boxes.shape


    # # Adding missing bounding boxes. Here, we define the minimum height space and minimum width space between the bounding boxes in a column and row respectively, in case of a missing bounding box.
    # # Add missing ROIs to the contours
    # new_contours = add_missing_rois(contours_sorted, space_height_threshold, space_width_threshold, max_cell_height_per_box, no_of_rows, no_of_columns, image_width)




    # Apply the same dot removal to the **binarized image**
    table_img_bin[dots_to_remove == 255] = 255  # Set dots to white

    # Ensure the image is strictly binary
    table_img_bin[table_img_bin > 127] = 255  # Set everything above 127 to white
    table_img_bin[table_img_bin <= 127] = 0   # Set everything below 127 to black


    # Save the binary image for use later in detecting text
    save_dir = os.path.join(transient_transcription_output_dir, station)
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    save_path_without_dots = os.path.join(save_dir, 'table_binarized_without_dots.jpg')
    cv2.imwrite(save_path_without_dots, table_img_bin)

    # Original image of table in binarizesd format without the dots
    table_binarized_without_dots_file = cv2.imread(save_path_without_dots)
    table_binarized_without_dots = table_binarized_without_dots_file.copy()



    # plt.imshow(table_binarized_without_dots)
    # plt.title("Filtered Image - Text Only - Without dots")
    # plt.show()



    # detected_table_cells = [new_contours, image_with_all_bounding_boxes, table_binarized_without_dots, table_original_image, full_detected_table_with_labels]

    # detected_table_cells = [contours_sorted, image_with_all_bounding_boxes, table_binarized_without_dots, table_original_image, full_detected_table_with_labels]

    table_for_overlaying_with_contours = table_img_bin.copy()
    # Convert the grayscale image to RGB to support colored bounding boxes
    table_for_overlaying_with_contours = cv2.cvtColor(table_for_overlaying_with_contours, cv2.COLOR_GRAY2RGB)
    detected_table_cells = [contours_sorted, image_with_all_bounding_boxes, table_for_overlaying_with_contours, table_original_image, full_detected_table_with_labels]

    # min_cell_width_threshold=int(config['TableAndCellDetection']['min_cell_width_threshold'])
    # max_cell_width_threshold=int(config['TableAndCellDetection']['max_cell_width_threshold'])
    # min_cell_height_threshold=int(config['TableAndCellDetection']['min_cell_height_threshold'])
    # max_cell_height_threshold=int(config['TableAndCellDetection']['max_cell_height_threshold'])

    ## FOR VISUALIZATION PURPOSES. Uncomment the lines below to plot the identified cells (contours/bounding boxes)
    # Make a copy of the original image to overlay contours without modifying the original
    table_img_bin_overlayed_with_contours = table_img_bin.copy()
    # Convert the grayscale image to RGB to support colored bounding boxes
    table_img_bin_overlayed_with_contours = cv2.cvtColor(table_img_bin_overlayed_with_contours, cv2.COLOR_GRAY2RGB)

    # Filter out smaller or larger bounding boxes from all the detected text contours. This is helpful to avoid overly large cells or small cells with no text. Remember to adjust these values based on the table structure in your specific case 
    filtered_contours = filter_contours(contours, min_cell_width_threshold, min_cell_height_threshold, max_cell_width_threshold, max_cell_height_threshold)

    logger.debug('Iterate over each contour in the new_contours list and draw bounding boxes')
    for contour in filtered_contours:
        if contour is not None and len(contour) > 0:
            x, y, w, h = cv2.boundingRect(contour)

            # # Adjust bounding box dimensions
            # increase_factor_width = 0.05
            # increase_factor_height = 0.25
            # x += int(w * increase_factor_width) # Increase width
            # y -= int(h * increase_factor_height) # Increase height
            # w -= int(w * increase_factor_width) # Decrease width a little to avoid vertical lines that may be transcribed as the number 1 yet they aren't a number
            # h += int(h * increase_factor_height * 2) # Increase height

            # Define increase factors for bounding box modification
            increase_factor_width = 0.20  # Increase width by 20%
            increase_factor_height = 0.2  # Increase height by 25%

            # Expand width while keeping it centered
            new_w = int(w * (1 + increase_factor_width))  # Increase width
            x = max(0, x - (new_w - w) // 2)  # Adjust x to keep center fixed

            # Expand height symmetrically
            new_h = int(h * (1 + increase_factor_height * 2))  # Increase height
            y = max(0, y - (new_h - h) // 2)  # Adjust y to keep center fixed

            # Ensure bounding box remains within valid image bounds
            x = max(0, x)
            y = max(0, y)
            w = max(1, new_w)  # Avoid zero or negative width
            h = max(1, new_h)  # Avoid zero or negative height
            
            # Draw the bounding box directly on the overlay image
            cv2.rectangle(table_img_bin_overlayed_with_contours, (x, y), (x + w, y + h), (255, 0, 0), 3)


    # Sort contours by y-coordinate
    contours_sorted = sorted(filtered_contours, key=lambda c: cv2.boundingRect(c)[1])


    ## STEP 6: REMOVE ANY NOISE (SUCH AS DOTS) FROM THE BINARIZED IMAGE OF THE TABLE TO PREPARE IT FOR THE NEXT TRANSCRIPTION MODULE
    # Apply the same dot removal to the **binarized image**
    table_img_bin[dots_to_remove == 255] = 255  # Set dots to white

    # Ensure the image is strictly binary
    table_img_bin[table_img_bin > 127] = 255  # Set everything above 127 to white
    table_img_bin[table_img_bin <= 127] = 0   # Set everything below 127 to black

    table_for_overlaying_with_contours = table_img_bin.copy()
    # Convert the grayscale image to RGB to support colored bounding boxes
    table_for_overlaying_with_contours = cv2.cvtColor(table_for_overlaying_with_contours, cv2.COLOR_GRAY2RGB)
    detected_table_cells = [contours_sorted, image_with_all_bounding_boxes, table_for_overlaying_with_contours, table_original_image, full_detected_table_with_labels]
    

    # # Display the image with bounding boxes using matplotlib
    # plt.figure(figsize=(50, 50))
    # plt.imshow(table_img_bin_overlayed_with_contours)
    # plt.axis('off')  # Hide axis
    # plt.show()


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

    # Here we use two methods to arrange the boundung boxes (as a double check): (1) Using the middle coordinates of the bounding boxes , and (2) Using the top coordinated of the boudning boxes.
    organize_methods = {  
        'Midpoint': organize_contours_midpoint,
        'Top': organize_contours_top
    }

    # organize_methods = {  
    #         'Midpoint': organize_contours_by_column,
    #         'Top': organize_contours_by_column
    #     }

    max_columns = no_of_columns
    max_rows = no_of_rows

    for method_name, organize_method in organize_methods.items():

        ## Create an Excel workbook and add a worksheet where the transcribed text will be saved
        wb = Workbook()
        ws = wb.active
        ws.title = 'OCR_Results'

        # Organize the contours
        # organized_rows = organize_method(new_contours, no_of_rows)

        organized_cols = organize_columns(new_contours, month_filename, max_columns)
        sorted_columns = sorted(organized_cols, key=lambda col: calculate_trimmed_mean([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] // 2 for c in col]))
        sorted_columns = remove_small_cells_in_column(sorted_columns)

        # Create a copy of the image for visualization
        image_sorted_cols = table_copy.copy()

        # Generate colors for 43 rows. For sorted row visualitation purposes
        colors = generate_random_colors(len(sorted_columns))  # Random colours for the maximum number of rows, such that each row has its own color for easy identification

        # Draw markers on the image before sorting
        draw_row_markers_and_boxes(image_sorted_cols, sorted_columns, colors)  # Green color for original order

        contours = [c for col in sorted_columns for c in col if c is not None]
        organized_rows = organize_contours_top(contours, None, max_rows)
        
        # Sorting the cell (bounding box) rows from first to last using trimmed mean of coordinates
        sorted_rows = sorted(organized_rows, key=lambda row: calculate_trimmed_mean([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] // 2 for c in row]))

        # Create a copy of the image for visualization
        image_sorted_rows = table_copy.copy()

        # Generate colors for 43 rows. For sorted row visualitation purposes
        colors = generate_random_colors(len(sorted_rows))  # Random colours for the maximum number of rows, such that each row has its own color for easy identification

        # Draw markers on the image before sorting
        draw_row_markers_and_boxes(image_sorted_rows, sorted_rows, colors)  # Green color for original order


        # Get the expected max rows (dynamically)
        # max_rows = get_max_rows_from_filename(month_filename)

        # Calculate how many rows need to be padded
        # missing_rows = 43 - max_rows  # Always ensuring 43 rows
        missing_rows = max_rows - len(sorted_rows)  # Calculate how many rows are missing to reach the maximum number of rows

        # If there are missing rows, pad `None` before the last two rows (totals & averages)
        if missing_rows > 0:
            # Ensure we don't disturb the last two rows
            sorted_rows = sorted_rows[:-2] + [[None]] * missing_rows + sorted_rows[-2:]

        sorted_rows = add_missing_boxes(sorted_rows, max_cell_width_threshold*0.8, max_cell_height_threshold*0.6, max_columns)  # Add missing boxes per row where there are gaps

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

        # Save image for inspection
        sorted_image_path = os.path.join(save_dir, f'sorted_cells_{station}_{month_filename}.jpg')
        cv2.imwrite(sorted_image_path, image_after_sorting)
        detected_table_cells = [sorted_rows, image_with_all_bounding_boxes, table_for_overlaying_with_contours, table_original_image, full_detected_table_with_labels]

    return detected_table_cells