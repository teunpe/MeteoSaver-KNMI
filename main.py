import os
import configparser
import multiprocessing as mp
from datetime import datetime
import glob
import cv2
import logging
import pickle
from tqdm.auto import tqdm
import numpy as np

from src.data_formatting_and_upload import data_formatting
from src.image_preprocessing_module import image_preprocessing
from src.quality_assessment_and_quality_control import qa_qc
from src_knmi.table_and_cell_detection_model_knmi import table_and_cell_detection
from src_knmi.utils.transcription import transcription
from src.validation import validate

from src_knmi.utils.preprocessing import deskew, read_img, image_preprocessing
from src_knmi.utils.table_detection import table_detection
from src_knmi.table_and_cell_detection_model_knmi import table_and_cell_detection

def main(file_path: str):

    # Set up logging
    filename = os.path.splitext(os.path.basename(file_path))[0]
    country, year, image = filename.split('_')[0], filename.split('_')[1], filename.split('_')[2]

    logger = logging.getLogger('meteosaver')
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    debughandler = logging.FileHandler(f'logs/{country}_{year}_{image}.log', mode='w')
    debughandler.setLevel(logging.DEBUG)
    debughandler.setFormatter(formatter)
    logger.addHandler(debughandler)

    infohandler = logging.StreamHandler()
    infohandler.setLevel(logging.INFO)
    infohandler.setFormatter(formatter)
    logger.addHandler(infohandler)
    logger.setLevel(logging.DEBUG)

    print("Setting up environment")
    # Module 1: Configuration
    # Load settings from user configurations. See configuration.ini file in this repository
    config = configparser.ConfigParser()
    # Adjust path to config.ini (since it's placed in the root directory)
    # config_file_path = os.path.join(os.path.dirname(
        # os.path.dirname(__file__)), 'configuration.ini')
    config.read('configs/base.ini')

    # Get run_mode and number of processors from configuration file
    run_mode = config['General']['run_mode']
    num_processors = int(config['General'].get('num_processors', 1))

    # Set up directories from the configuration file
    full_datadir = config['Directories']['full_datadir'] # Directory to all the images/scans of the hydroclimatic data sheets
    pre_QA_QC_transcribed_hydroclimate_data_dir = config['Directories']['pre_QA_QC_transcribed_hydroclimate_data_dir'] # Directory where pre-QA/QC transcribed data is stored
    post_QA_QC_transcribed_hydroclimate_data_dir = config['Directories']['post_QA_QC_transcribed_hydroclimate_data_dir'] # Directory where post-QA/QC transcribed data is stored
    final_refined_daily_hydroclimate_data_dir = config['Directories']['final_refined_daily_hydroclimate_data_dir'] # Directory for the final refined daily hydroclimate data (after all quality checks)
    manually_transcribed_data_dir = config['Directories']['manually_transcribed_data_dir'] # Directory for manually transcribed data (used for validation)
    validation_dir = config['Directories']['validation_dir'] # Directory for validation results comparing manually transcribed and the MeteoSaver transcribed data 
    transient_transcription_output_dir = config['Directories']['transient_transcription_output_dir'] # Directory to store transient transcription output during processing
    metadata_file_path = config['Directories']['metadata_file_path'] # Directory for all the stations metadata

    # Get all folder names (Station Numbers) within full_datadir
    all_stations = [folder for folder in os.listdir(full_datadir) 
                    if os.path.isdir(os.path.join(full_datadir, folder))]
    
    # Set up configuations for a specific station
    station = 'knmi'
    pre_QA_QC_transcribed_hydroclimate_data_dir_station = os.path.join(pre_QA_QC_transcribed_hydroclimate_data_dir, station)
    month_filename = file_path
    
    # Check if saved detection pickle file exists
    if os.path.exists(os.path.join(f"saved_detections/{station}_{month_filename.split('/')[-1].replace('.jpg', '')}_detected_table_and_cells.pkl")):
        with open(os.path.join(f"saved_detections/{station}_{month_filename.split('/')[-1].replace('.jpg', '')}_detected_table_and_cells.pkl"), 'rb') as f:
            detected_table_and_cells = pickle.load(f)
    else:
        logger.info("Preprocessing...")
        conf_file = file_path.replace('.jpg', '.ini')
        if not os.path.exists(conf_file):
            # remove trailing number from filename to get default config for country
            
            conf_file = conf_file.replace(filename, f"{country}_{year}")
            if os.path.exists(conf_file):
                logger.info(f"Conf file not found, using default for country: {conf_file}")
            else:
                for y in range(int(year)-1, 1900, -1):
                    conf_file = f'data/00_knmi_images/trainset/{country}_{y}.ini'
                    if os.path.exists(conf_file):
                        config.read(conf_file)
                        logger.info(f'Using config file from year {y} for {country} {year}')
                        break
            config.read(conf_file)
            no_of_rows = config.getint('Cell_detection', 'no_of_rows')
            month = int(filename.split('_')[2])/2
            # get days in month based on month (not accounting for leap years)
            if month in [1, 3, 5, 7, 8, 10, 12]:
                no_of_rows -= 0 # including header
            elif month in [4, 6, 9, 11]:
                no_of_rows -= 1 # including header
            elif month == 2:
                no_of_rows -= 3 # including header

        else:
            config.read(conf_file)
            no_of_rows = config.getint('Cell_detection', 'no_of_rows')
        no_of_columns = config.getint('Cell_detection', 'no_of_columns')
        # img_path = f'/home/teun/knmi/data-rescue-internship-teun/data/jpg/{country}/{year}/{file_path}'
        img_path = f'data/00_knmi_images/testset/{country}_{year}_{image}.jpg'
        img = read_img(img_path)
        config.read(conf_file)
        blocksize = config.getint('Table_detection', 'blocksize')
        C = config.getint('Table_detection', 'C')
        table_width = config.getint('Table_detection', 'w')
        table_height = config.getint('Table_detection', 'h')
        table_x_offset = config.getint('Table_detection', 'x')
        table_y_offset = config.getint('Table_detection', 'y')
        reference_word = config['Table_detection']['reference']
        min_cell_width_threshold = config.getint('Cell_detection', 'min_cell_width')
        max_cell_width_threshold = config.getint('Cell_detection', 'max_cell_width')
        min_cell_height_threshold = config.getint('Cell_detection', 'min_cell_height')
        max_cell_height_threshold = config.getint('Cell_detection', 'max_cell_height')
        no_of_columns = config.getint('Cell_detection', 'no_of_columns')

        # Binarize and deskew image
        image_in_grayscale, binarized_image, original_image = image_preprocessing(img, blocksize=blocksize, C=6, skew=True)

        # Detect table based on reference word location
        size = (table_width, table_height)
        table_offset=(table_x_offset, table_y_offset)
        initial_offset=(1000, 500) # Remove left and top part of the image to search for the reference word
        reference_word=reference_word
        x, y, w, h, df = table_detection(binarized_image, reference_word=reference_word, size=size, 
                                        initial_offset=initial_offset, table_offset=table_offset)

        if x is not None:
            table_original_image = original_image[y:y + h, x:x + w]
            image_in_grayscale, binarized_image, original_image = image_preprocessing(table_original_image, cutoff_hsv=125, skew=False, blur=True, blocksize=15, C=3)
            
            table_img_bin = binarized_image
            table_original_image = original_image
            full_detected_table_with_labels = binarized_image
            # cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
            # cv2.imshow('custom window', table_img_bin)
            # cv2.resizeWindow('custom window', 1080, 720)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
        else:
            df.head()
            return


        print("Running Table and Cell Detection Model...")
        detected_table_and_cells = table_and_cell_detection(image_in_grayscale, binarized_image, original_image, station, month_filename, transient_transcription_output_dir,
                                                    clip_up = int(config['TableAndCellDetection']['clip_up']), 
                                                    clip_down = int(config['TableAndCellDetection']['clip_down']),
                                                    clip_left = int(config['TableAndCellDetection']['clip_left']),
                                                    clip_right = int(config['TableAndCellDetection']['clip_right']),
                                                    max_table_width = int(config['TableAndCellDetection']['max_table_width']),
                                                    max_table_height = int(config['TableAndCellDetection']['max_table_height']),
                                                    min_cell_width_threshold=min_cell_width_threshold,
                                                    max_cell_width_threshold=max_cell_width_threshold,
                                                    min_cell_height_threshold=min_cell_height_threshold,
                                                    max_cell_height_threshold=max_cell_height_threshold,
                                                    space_height_threshold=int(config['TableAndCellDetection']['space_height_threshold']), 
                                                    space_width_threshold=int(config['TableAndCellDetection']['space_width_threshold']), 
                                                    max_cell_height_per_box=int(config['TableAndCellDetection']['max_cell_height_per_box']), 
                                                    no_of_rows=no_of_rows,
                                                    no_of_columns=no_of_columns)
        

        # save result to pickle file
        with open(os.path.join(f"saved_detections/{station}_{month_filename.split('/')[-1].replace('.jpg', '')}_detected_table_and_cells.pkl"), 'wb') as f:
            pickle.dump(detected_table_and_cells, f)

    print("Running Transcription...")
    # Module 4: Transcription
    logger.info("Step 3: Transcribing values using OCR/HTR")
    start_time = datetime.now()
    ocr_model = config['Transcription']['ocr_model'] # Selected OCR/HTR model
    # Incase of Tesseract
    # Ensure that the tesseract path is set correctly for your local system
    tesseract_path = config['Transcription']['tesseract_path']
    # Set TESSDATA_PREFIX to the system's tessdata directory (for system-wide language files)
    system_tessdata_dir = config['Transcription']['system_tessdata_dir']
    os.environ["TESSDATA_PREFIX"] = system_tessdata_dir

    
    transcribed_table = transcription(
        detected_table_and_cells, ocr_model, tesseract_path, transient_transcription_output_dir, 
        pre_QA_QC_transcribed_hydroclimate_data_dir_station, station, month_filename,
        no_of_rows=int(config['TableAndCellDetection']['no_of_rows']),
        no_of_columns=int(config['TableAndCellDetection']['no_of_columns']),
        no_of_rows_including_headers=int(config['TableAndCellDetection']['no_of_rows_including_headers']),
        max_columns=21, max_rows=37,
        max_cell_width_threshold=int(config['TableAndCellDetection']['max_cell_width_threshold']),
        max_cell_height_threshold=int(config['TableAndCellDetection']['max_cell_height_threshold'])
    )

    end_time = datetime.now()

    print(f'Duration of transcribing: {end_time - start_time}')

if __name__ == "__main__":
    # for country in ['st-eustatius', 'curacao']:
    #     for year in tqdm([1911], desc='Years'):
    #         for image in tqdm(np.arange(2,24,2), desc='Images'):
    #             # if country == 'st-eustatius' and image == 12:
    #             #     continue
    #             if os.path.exists(f'results/05_transient_transcription_output/knmi/full_table_{country}_{year}_{image:04d}.jpg.jpg'):
    #                 print(f'Skipping {country} {year} image {image:04d}, already exists...')
    #                 continue
    #             print(f'Processing {country} {year} image {image:04d}...')
    #             try:
    #                 main(f'{country}_{year}_{image:04d}.jpg')
    #             except Exception as e:
    #                 print(e)
    #                 continue
    for filename in tqdm(glob.glob('data/00_knmi_images/testset/*.jpg'), desc='Processing images'):
        print(f'Processing {filename}...')
        try:
            main(filename)
        except Exception as e:
            print(e)
            continue