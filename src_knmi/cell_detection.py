import cv2
import numpy as np
import os

from src.table_and_cell_detection_model import filter_contours


def cell_detection(
        image_without_lines_noise_removed, table_img_bin,
        table_original_image, full_detected_table_with_labels,
        station, transient_transcription_output_dir, 
        min_cell_width_threshold, 
        min_cell_height_threshold, max_cell_width_threshold, 
        max_cell_height_threshold, 
        ): 
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

    # Step 2: Identify large text components (potential numbers)
    text_components = []
    dots_to_remove = np.zeros_like(labels)  # Mask for dots to be removed

    for i in range(1, num_labels):
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

    # Step 3: Process small dots and keep only those near text
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]

        # Skip already kept text components
        if area > 100 and aspect_ratio < ASPECT_RATIO_THRESHOLD and h > HEIGHT_THRESHOLD:
            continue  # Skip large text components

        # Check if this small dot is close to any number
        keep_dot = False
        for tx, ty, tw, th in text_components:
            # Compute distance from dot center to the number bounding box
            dot_center_x, dot_center_y = x + w // 2, y + h // 2
            if (tx - PROXIMITY_THRESHOLD <= dot_center_x <= tx + tw + PROXIMITY_THRESHOLD and
                ty - PROXIMITY_THRESHOLD <= dot_center_y <= ty + th + PROXIMITY_THRESHOLD):
                keep_dot = True
                # Store the y-position and x-limits of dots near text
                horizontal_lines.append((dot_center_y, x, x + w))  # (y-position, x_start, x_end)
                break  # No need to check other numbers

        if keep_dot:
            filtered_image[labels == i] = 255  # Keep dots near numbers
        # else:
        #     dots_to_remove[labels == i] = 255  # Mark dots for removal

    # plt.imshow(filtered_image, cmap="gray")
    # plt.title("Second Filtered Image - Only Dots Near Text - No Lines")
    # plt.show()
    

    # Step 2: Use **horizontal dilation** to merge digits within numbers, but prevent full merging
    kernel_to_remove_gaps = np.ones((1, 6), np.uint8)  # Horizontal merging kernel
    image_with_number_blobs = cv2.dilate(filtered_image, kernel_to_remove_gaps, iterations=5)  # Controlled dilation

    # Step 3: Apply **morphological closing** to ensure numbers remain compact blobs
    rect_kernel = np.ones((1, 3), np.uint8)   # was 3 initially. in prev step was 1
    image_with_word_blobs = cv2.morphologyEx(image_with_number_blobs, cv2.MORPH_CLOSE, rect_kernel, iterations=1)


    # plt.imshow(image_with_word_blobs, cmap = 'gray') # figure showing detected table image with horizintal and vertical lines removed.
    # plt.title('horizontal dilation')
    # plt.show() 


    # Define a horizontal erosion kernel
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))  # (height, width) to break horizontal merges. had it at 1,5 initially 
    # Apply erosion to break the connection between digits
    image_with_word_blobs = cv2.erode(image_with_word_blobs, horizontal_kernel, iterations=2)  # Increase iterations if still connected. Had 2 iteratons initially
    # plt.imshow(image_with_word_blobs, cmap = 'gray') # figure showing detected table image with horizintal and vertical lines removed.
    # plt.title('applied horizontal_kernel')
    # plt.show() 

    # ** delete small horizontal lines.
    # Create a blank mask for horizontal lines
    horizontal_line_mask = np.zeros_like(image_with_word_blobs)

    # Draw **thin** horizontal lines at detected y-positions (but not subtracting yet)
    for y, x_start, x_end in horizontal_lines:
        cv2.line(horizontal_line_mask, (x_start - 15, y), (x_end + 15, y), 255, thickness=3)  # Initially very thin

    # # # Show the initial thin line mask
    # # plt.imshow(horizontal_line_mask, cmap="gray")
    # # plt.title("Initial Thin Line Mask")
    # # plt.show()

    # **Step 3: Subtract the dilated lines from the filtered image**
    image_with_word_blobs = cv2.subtract(image_with_word_blobs, horizontal_line_mask)

    # # Show the final image after subtraction
    # plt.imshow(image_with_word_blobs, cmap="gray")
    # plt.title("Final Image After Subtracting Horizontal Lines")
    # plt.show()



    ####*********what if i did this step above now with (1,3) dilation. In otherwords in both direction
    image_with_word_blobs = cv2.dilate(image_with_word_blobs, np.ones((1, 6), np.uint8), iterations=2) # (height, width) # initialy (1,3) and 1 iteration
    # plt.imshow(image_with_word_blobs, cmap = 'gray') 
    # plt.title('final horizontal dilation to increase width of blobs')
    # plt.show() 

    # **Step 2: Apply Controlled Erosion to Fix Over-Merging**
    erosion_kernel = np.ones((1, 3), np.uint8)  # Smaller horizontal erosion kernel
    image_with_word_blobs = cv2.erode(image_with_word_blobs, erosion_kernel, iterations=1)  # Controlled shrinking

    # plt.imshow(image_with_word_blobs, cmap="gray")
    # plt.title("After Horizontal Erosion (Fix Over-Merging)")
    # plt.show()

    # **Step 2: Find Wide Blobs That Need Stronger Erosion**
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image_with_word_blobs, connectivity=8)

    for i in range(1, num_labels):  # Ignore background (label 0)
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        if w > 200:  # **If the blob is too wide, apply stronger erosion**
            erosion_kernel = np.ones((3, 10), np.uint8)  # Slightly stronger erosion for wide blobs
            roi = image_with_word_blobs[y:y+h, x:x+w]
            eroded_roi = cv2.erode(roi, erosion_kernel, iterations=3)  # More iterations for wider blobs'
            
            # **Apply vertical recovery dilation to restore eroded thickness**
            dilation_kernel = np.ones((2, 1), np.uint8)  # Small dilation to restore height
            recovered_roi = cv2.dilate(eroded_roi, dilation_kernel, iterations=3)  # Restore vertical thickness
            
            image_with_word_blobs[y:y+h, x:x+w] = recovered_roi  # Replace only this region
        
        if h > 50: # ** If the blob is too tall, apply stronger erosion along the horizontal dimension**
            # erosion_kernel_vert = np.ones((min(h // 15, 12), 6), np.uint8)  # Adaptive vertical erosion # Initially had this: np.ones((min(h // 20, 12), 3), np.uint8)
            erosion_kernel_vert = np.ones((1, 13), np.uint8)  # Slightly stronger erosion for wide blobs # Imitially had (10,3), BUT (1,10) improved it
            roi = image_with_word_blobs[y:y+h, x:x+w]
            eroded_roi = cv2.erode(roi, erosion_kernel_vert, iterations=3)  
            
            # **Restore horizontal thickness**
            dilation_kernel_vert = np.ones((1, 6), np.uint8)  # initially had (1,6)
            recovered_roi = cv2.dilate(eroded_roi, dilation_kernel_vert, iterations=3)  

            image_with_word_blobs[y:y+h, x:x+w] = recovered_roi 

    
    # plt.imshow(image_with_word_blobs, cmap="gray")
    # plt.title("After Adaptive Erosion (Fixing Over-Merging)")
    # plt.show()

    
    

    ## Using contours in order to detect text in the table after removing the vertical and horizontal lines
    # Assuming 'table' is your input image in BGR format
    # result = cv2.findContours(image_without_lines_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.findContours(image_with_word_blobs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = result[0]
    # Original image of table in binarizesd format
    image_with_all_bounding_boxes = table_img_bin
    # table_binarized = image_with_all_bounding_boxes.copy()
    

    # # Plots only for visualization purposes. Uncomment the lines below to show the different steps
    
    # plt.imshow(image_without_lines, cmap = 'gray') # figure showing detected table image with horizintal and vertical lines removed.
    # plt.show() 

    # plt.imshow(image_without_lines_noise_removed, cmap = 'gray') # figure showing detected table image with horizintal and vertical lines removed.
    # plt.show() 

    # plt.imshow(image_with_word_blobs, cmap = 'gray') # figure showing text blobs on the detected table image with horizintal and vertical lines removed.
    # plt.show()

    # plt.imshow(image_without_lines_2, cmap = 'gray') # figure showing text blobs on the detected table image with horizintal and vertical lines removed.
    # plt.show()

    # plt.imshow(detected_table_cells[4], cmap = 'gray') # unclipped detected table
    # plt.show()

    # plt.imshow(detected_table_cells[1]) # clipped detected table
    # plt.show()

    ## FOR VISUALIZATION PURPOSES. Uncomment the lines below to plot the identified cells (contours/bounding boxes)
    # Make a copy of the original image to overlay contours without modifying the original
    table_img_bin_overlayed_with_contours = table_img_bin.copy()
    # Convert the grayscale image to RGB to support colored bounding boxes
    table_img_bin_overlayed_with_contours = cv2.cvtColor(table_img_bin_overlayed_with_contours, cv2.COLOR_GRAY2RGB)


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
            increase_factor_width = 0.07  # Increase width by 20%
            increase_factor_height = 0.25  # Increase height by 25%

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

    # # Display the image with bounding boxes using matplotlib
    # plt.imshow(table_img_bin_overlayed_with_contours)
    # plt.axis('off')  # Hide axis
    # plt.show()

    # Sort contours by y-coordinate
    contours_sorted = sorted(filtered_contours, key=lambda c: cv2.boundingRect(c)[1])

    # Get the dimensions of the loaded image. Here, particulary the image/table width is very important for the column placement of cells/bounding boxes
    image_height, image_width, image_channels = image_with_all_bounding_boxes.shape


    # # Adding missing bounding boxes. Here, we define the minimum height space and minimum width space between the bounding boxes in a column and row respectively, in case of a missing bounding box.
    # # Add missing ROIs to the contours
    # new_contours = add_missing_rois(contours_sorted, space_height_threshold, space_width_threshold, max_cell_height_per_box, no_of_rows, no_of_columns, image_width)
    

    # ## FOR VISUALIZATION PURPOSES. Uncomment the lines below to plot the identified cells (contours/bounding boxes)
    # # Make a copy of the original image to overlay contours without modifying the original
    # table_img_bin_overlayed_with_contours = table_img_bin.copy()
    # # Convert the grayscale image to RGB to support colored bounding boxes
    # table_img_bin_overlayed_with_contours = cv2.cvtColor(table_img_bin_overlayed_with_contours, cv2.COLOR_GRAY2RGB)


    # # Iterate over each contour in the new_contours list and draw bounding boxes
    # for contour in new_contours:
    #     if contour is not None and len(contour) > 0:
    #         x, y, w, h = cv2.boundingRect(contour)

    #         # Adjust bounding box dimensions
    #         increase_factor_width = 0.2
    #         increase_factor_height = 0.3
    #         # x += int(w * increase_factor_width) # Increase width
    #         # y -= int(h * increase_factor_height) # Increase height
    #         # w -= int(w * increase_factor_width) # Decrease width a little to avoid vertical lines that may be transcribed as the number 1 yet they aren't a number
    #         # h += int(h * increase_factor_height * 2) # Increase height

    #         # Expand width
    #         new_w = int(w * (1 + increase_factor_width))  # Increase width
    #         x = x - (new_w - w) // 2  # Center the new width

    #         # Expand height
    #         new_h = int(h * (1 + increase_factor_height * 2))  # Increase height on both sides
    #         y = y - (new_h - h) // 2  # Center the new height
            
    #         # # Draw the bounding box directly on the overlay image
    #         # cv2.rectangle(table_img_bin_overlayed_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 4)

    #         # Draw the updated bounding box
    #         cv2.rectangle(table_img_bin_overlayed_with_contours, (x, y), (x + new_w, y + new_h), (0, 255, 0), 2)

    # # Remove contours that are None or empty
    # new_contours = [contour for contour in new_contours if contour is not None and len(contour) > 0]

    # # Display the image with bounding boxes using matplotlib
    # plt.imshow(table_img_bin_overlayed_with_contours)
    # plt.axis('off')  # Hide axis
    # plt.show()


    # # Save the binary image for use later in detecting text
    # save_dir = os.path.join(transient_transcription_output_dir, station)
    # os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    # save_path_without_vertical_lines = os.path.join(save_dir, 'table_binarized_without_vertical_lines.jpg')
    # cv2.imwrite(save_path_without_vertical_lines, table_img_bin)

    # # Original image of table in binarizesd format without the dots
    # table_binarized_without_vertical_lines_file = cv2.imread(save_path_without_vertical_lines)
    # table_binarized_without_vertical_lines = table_binarized_without_vertical_lines_file.copy()

    # plt.imshow(table_binarized_without_vertical_lines)
    # plt.title("Filtered Image - Text Only - Without vertical lines")
    # plt.show()



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

    detected_table_cells = [contours_sorted, image_with_all_bounding_boxes, table_binarized_without_dots, table_original_image, full_detected_table_with_labels]

    

    return detected_table_cells