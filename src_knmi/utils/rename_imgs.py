import os

for root, dirs, files in os.walk("data/00_knmi_images"):
    station = root.split("/")[-1]
    for file in files:
        if file.endswith(".jpg"):
            old_path = os.path.join(root, file)
            _, year, number_filetype = file.split("_")
            number, _ = number_filetype.split(".")
            month = str(int(int(number)/2)).zfill(2)
            print(station, year, month)
            new_name = f'{station}_{year}{month}_SF.jpg'
            new_path = os.path.join(root, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} to {new_path}")
        else:
            print(f"Skipped: {file}")