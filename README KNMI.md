# KNMI adjustments for MeteoSaver

Here we describe the adjustments made to the MeteoSaver code to run it on our tables.

## src_knmi

The folder `src_knmi` contains the files of the MeteoSaver source that were changed. The files correspond to those used in the original MeteoSaver.
- `preprocessing.py`: performs preprocessing; binarization and deskew.
- `rename_imgs.py`: old functionality to rename images to MeteoSaver format, most likely not needed.
- `table_detection.py`: table detection using our OCR technique.
- `transcription.py`: adjusted MeteoSaver transcription. MeteoSaver adds missing cells during the transcription step, we have moved this to the cell detection step. Line 382 selects the model, which we replace with our finetuned models as needed. We also disable the Midpoint cell detection, using only the tops of contours for the cell clustering.
- `table_and_cell_detection_model_knmi.py`: adjusted MeteoSaver cell detection. Added extra column operation removing cells that are much less wide from a column. Adjusted some values in `add_missing_boxes`. Removed the table detection (moved to `table_detection.py`).

## single_sheet_setup.ipynb
`single_sheet_setup.ipynb` helps with setting the parameters for a table. This means setting the parameters for preprocessing, and for the table detection. These parameters can be added to a `.ini` file in `data/00_knmi_images/*/` with the name of the table in question, or just the country and year for a `.ini` file that works for several tables. When running the pipeline it first looks for the file-specific `.ini`, but if it doesn't exist it selects the `.ini` for the country and year instead.

## main.py
`main.py` runs the transcription on the specified set of tables. It first looks for the best `.ini` as described above, then runs the table and cell detection followed by the transcription. It saves the cell detection and will load it from the save if it exists, speeding up the process during testing of transcription. 

## Output
Results are saved in `results/05_transient_transcription_output/knmi`. It will have a folder for each table containing the Excel sheet with the transcription. The `.zip` files contain the clipped cells. The files starting with `full_table` contain an image with bounding boxes for each cell. Test set transcriptions are saved in `results/05_transient_transcription_output/knmi/data/00_knmi_images/testset`, and cell detection images in `results/05_transient_transcription_output/knmi/full_table_data/00_knmi_images/testset`.