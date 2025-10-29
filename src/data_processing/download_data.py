import gdown
import os

ORIGINAL_DATA_ID = "1utmqvJlvnEGHfz6bHs_fQg0fgby0leA"
PROCESSED_DATA_ID = "1bE3AkPbOsH_9cfrIN9QZiw7zAGGu4FW4" # Data with no lowercasing 0.879 F1-Score.
PROCESSED_DATA_ID = "1cn4bX4l2ESRvQDN-Gd_UfS-uY6Rzmf-F" # Data with lowercasing 0.8813 F1-Score.
OUTPUT_PATH = "YOUR_PATH"

def download_gdrive_file(file_id, output_path):
    """
    Downloads a file from Google Drive using its file ID.

    Args:
        file_id (str): The ID of the file to download.
        output_path (str): The path to save the downloaded file.
    """
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
        print(f"Successfully downloaded file to: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":

    google_drive_file_id = ORIGINAL_DATA_ID

    output_filename = OUTPUT_PATH 

    download_gdrive_file(google_drive_file_id, output_filename)