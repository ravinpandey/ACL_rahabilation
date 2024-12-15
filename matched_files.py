import os
import argparse
import re
import shutil
import pandas as pd

def load_xlsx_data(file_path):
    """
    Load the Excel file and return a DataFrame.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Data from the Excel file.
    """
    return pd.read_excel(file_path)

def filter_and_save_files(directory, good_dir, less_good_dir, xlsx_file):
    """
    Match filenames with numeric part from Excel file and save to corresponding directories.

    Args:
        directory (str): Path to the directory to search for files.
        good_dir (str): Path to the directory to save 'good' files.
        less_good_dir (str): Path to the directory to save 'less good' files.
        xlsx_file (str): Path to the Excel file.
    """
    # Load data from Excel file
    data = load_xlsx_data(xlsx_file)

    # Ensure output directories exist
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(less_good_dir, exist_ok=True)

    # Extract numeric parts from the afile column
    afile_data = data[['aFile', 'good', 'less good']]
    afile_data['digit'] = afile_data['aFile'].str.extract(r'(\d+)')

    for root, _, files in os.walk(directory):
        for file in files:
            # Extract numeric part from the filename
            match = re.search(r'(\d+)', file)
            if match:
                file_digit = match.group(1)

                # Check if this digit exists in the Excel data
                row = afile_data[afile_data['digit'] == file_digit]
                if not row.empty:
                    row = row.iloc[0]

                    # Determine where to save the file
                    if row['good'] == 1:
                        shutil.copy(os.path.join(root, file), good_dir)
                    elif row['less good'] == 1:
                        shutil.copy(os.path.join(root, file), less_good_dir)

if __name__ == "__main__":
    # Set up argparse
    parser = argparse.ArgumentParser(description="Filter and save files based on matching numeric parts with Excel data.")
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory to search for files."
    )
    parser.add_argument(
        "good_directory",
        type=str,
        help="Path to the directory to save 'good' files."
    )
    parser.add_argument(
        "less_good_directory",
        type=str,
        help="Path to the directory to save 'less good' files."
    )
    parser.add_argument(
        "xlsx_file",
        type=str,
        help="Path to the Excel file containing filtering criteria."
    )

    # Parse the arguments
    args = parser.parse_args()
    directory_path = args.directory
    good_directory = args.good_directory
    less_good_directory = args.less_good_directory
    xlsx_file_path = args.xlsx_file

    # Filter and save files
    try:
        filter_and_save_files(directory_path, good_directory, less_good_directory, xlsx_file_path)
        print("Files have been filtered and saved successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
