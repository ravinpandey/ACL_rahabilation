import os
import argparse
import re
import shutil

def filter_p2d_and_text_files(directory, output_directory):
    """
    Recursively search for files containing '_p2d' in their filenames and further filter those
    containing '_A' or '_ND', excluding '_D' and '_NA', then copy them to another directory.
    
    Args:
        directory (str): Path to the root directory to search.
        output_directory (str): Path to the directory to save filtered files.

    Returns:
        list: List of matching file paths.
    """
    filtered_files = []

    # Define the patterns
    p2d_pattern = re.compile(r".*_p2d.*")
    specific_pattern = re.compile(r".*(_A|_ND)(?!.*(_D|_NA)).*")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file matches the '_p2d' pattern
            if p2d_pattern.match(file):
                # Further filter files containing '_A' or '_ND' but not '_D' or '_NA'
                if specific_pattern.match(file):
                    file_path = os.path.join(root, file)
                    filtered_files.append(file_path)

                    # Copy the file to the output directory
                    shutil.copy(file_path, output_directory)

    return filtered_files

if __name__ == "__main__":
    # Set up argparse
    parser = argparse.ArgumentParser(description="Filter files containing '_p2d' and then '_A' or '_ND' (excluding '_D' and '_NA'), and save them to another directory.")
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory to search for files."
    )
    parser.add_argument(
        "output_directory",
        type=str,
        help="Path to the directory to save filtered files."
    )

    # Parse the arguments
    args = parser.parse_args()
    directory_path = args.directory
    output_directory = args.output_directory

    # Filter files and save results
    try:
        filtered_files = filter_p2d_and_text_files(directory_path, output_directory)
        print("Filtered files saved to:", output_directory)
        for file in filtered_files:
            print(file)
    except Exception as e:
        print(f"An error occurred: {e}")
