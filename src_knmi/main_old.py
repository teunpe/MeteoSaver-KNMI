import os
import configparser
import glob

from src.image_preprocessing_module import image_preprocessing
from src_knmi.utils.logging import setup_logging
from src_knmi.table_detection import table_detection
from src_knmi.cell_detection import cell_detection 

logger = setup_logging()
STN_NUMBER = 72

# Module 1: Configuration
# Load settings from user configurations. See configuration.ini file in this repository
config = configparser.ConfigParser()
# Adjust path to config.ini (since it's placed in the root directory)
config_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configuration.ini')
config.read(config_file_path)

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

def main(station: int):
    datadir = os.path.join(full_datadir, station)
    pre_QA_QC_transcribed_hydroclimate_data_dir_station = os.path.join(pre_QA_QC_transcribed_hydroclimate_data_dir, station)
    post_QA_QC_transcribed_hydroclimate_data_dir_station = os.path.join(post_QA_QC_transcribed_hydroclimate_data_dir, station)
    
    # Ensure directories exist
    os.makedirs(pre_QA_QC_transcribed_hydroclimate_data_dir_station, exist_ok=True)
    os.makedirs(post_QA_QC_transcribed_hydroclimate_data_dir_station, exist_ok=True)
    
    # Search for all files in the `datadir` that start with the station number followed by an underscore ('_') and any other characters, and return a list of all matching file paths
    station_data = glob.glob(os.path.join(datadir, f"{station}_*")) 
    # OPTIONAL (Comment these two lines below if unnecessary): Filter the station_data and filenames to include only files with 'SF' in their names. Here the SF for standard format for sheets. This to filter our sheets with HD in their name, which stands for hand-drawn format, in their name as these were not standard formatted sheets and manually drawn by the observer.
    filenames = [os.path.basename(file) for file in station_data if 'SF' in os.path.basename(file)]
    for month in range(len(filenames)):
        month_data = station_data[month]
        month_filename = filenames[month]

        # Perform Pre-processing, Transcription, QA/QC, and Post-processing
        try:
            # Module 2: Image pre-processing
            image_in_grayscale, binarized_image, original_image = image_preprocessing(month_data)

            # Module 3: Table and cell detection 
            tables = table_detection(binarized_image)
            image_without_lines_noise_removed = tables[0]
            table_img_bin = tables[1]
            table_original_image = tables[2]
            full_detected_table_with_labels = tables[3]

            min_cell_width_threshold=int(config['TableAndCellDetection']['min_cell_width_threshold']),
            max_cell_width_threshold=int(config['TableAndCellDetection']['max_cell_width_threshold']),
            min_cell_height_threshold=int(config['TableAndCellDetection']['min_cell_height_threshold']),
            max_cell_height_threshold=int(config['TableAndCellDetection']['max_cell_height_threshold']),

            detected_table_cells = cell_detection(
                image_without_lines_noise_removed, table_img_bin,
                table_original_image, full_detected_table_with_labels,
                station, transient_transcription_output_dir, 
                min_cell_width_threshold, 
                min_cell_height_threshold, max_cell_width_threshold, 
                max_cell_height_threshold
                )
            
        except Exception as e:
            print(f"Error processing {month_filename}: {e}")
            continue
    pass

if __name__ == "__main__":
    main(STN_NUMBER)