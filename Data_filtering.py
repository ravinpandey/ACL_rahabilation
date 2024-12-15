import pandas as pd
import numpy as np
import os
import argparse


def process_file(file_path):
    """
    Process a single file to extract and transform data, adding trial_number and jump_type columns.
    """
    print(f"Processing file: {file_path}")

    try:
        # Load the data
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Skip rows until the first value of a row starts with 1
        data_lines = []
        for line in lines:
            values = line.split()
            if values and values[0] == '1':  # Check if the first value is '1'
                data_lines.append(line)
                break  # Found the starting point, break the loop

        if not data_lines:
            print(f"No data starting with '1' found in {file_path}")
            return None

        # Append remaining lines after the starting row
        start_index = lines.index(data_lines[-1])  # Get the index of the starting line
        data_lines.extend(lines[start_index + 1:])

        # Convert the data into a DataFrame
        data = pd.DataFrame([list(map(float, line.split())) for line in data_lines])

        # Define variable names
        variables = [
            "ankle_mom_X", "ankle_mom_Y", "ankle_mom_Z",
            "foot_X", "foot_Y", "foot_Z",
            "hip_mom_X", "hip_mom_Y", "hip_mom_Z",
            "hip_X", "hip_Y", "hip_Z",
            "knee_mom_X", "knee_mom_Y", "knee_mom_Z",
            "knee_X", "knee_Y", "knee_Z",
            "pelvis_X", "pelvis_Y", "pelvis_Z",
            "thorax_X", "thorax_Y", "thorax_Z"
        ]

        # Total number of variables (24 variables per row)
        num_variables = len(variables)

        # Initialize the resulting DataFrame
        final_data = []

        # Iterate over each frame's data
        for frame_number, row in data.iterrows():
            row_data = row.values[1:]  # Skip the frame number column
            total_values = len(row_data)  # Total values in the frame
            rows_per_frame = total_values // num_variables  # Divide into chunks of 24

            # Reshape the data into rows_per_frame x num_variables
            reshaped_data = row_data.reshape(rows_per_frame, num_variables)

            # Create a DataFrame for this frame's reshaped data
            frame_df = pd.DataFrame(reshaped_data, columns=variables)
            frame_df["Frame"] = int(row.values[0])  # Add frame number as a column

            # Append to the final data
            final_data.append(frame_df)

        # Concatenate all frames' data into a single DataFrame
        result = pd.concat(final_data, ignore_index=True)

        # Add jump_type column alternating between 'lat' and 'med'
        jump_types = ['lat', 'med']
        result['jump_type'] = [jump_types[i % 2] for i in range(len(result))]

        # Add trial_number column
        # Iterate over unique frames and create the trial number
        trial_numbers = []
        filename = os.path.basename(file_path).split(".")[0]  # Extract filename without extension
        for frame in result['Frame'].unique():
            frame_data = result[result['Frame'] == frame]
            # Alternate trial numbers (e.g., filename_1, filename_1, filename_2, filename_2)
            for trial_idx in range(len(frame_data) // 2):
                trial_numbers.extend([f"{filename}_{trial_idx + 1}"] * 2)

        result['trial_number'] = trial_numbers

        # Reorganize the columns
        result = result[['Frame', 'trial_number', 'jump_type'] + variables]

        return result

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def process_directories(input_dir1, input_dir2):
    """
    Process all `.txt` files in two directories and save the resulting DataFrames as CSV files.
    """
    for input_dir in [input_dir1, input_dir2]:
        print(f"Processing directory: {input_dir}")

        if not os.path.isdir(input_dir):
            print(f"Directory not found: {input_dir}")
            continue

        for file_name in os.listdir(input_dir):
            if file_name.endswith(".txt"):  # Process only .txt files
                file_path = os.path.join(input_dir, file_name)
                if os.path.isfile(file_path):  # Check if it is a file
                    processed_data = process_file(file_path)

                    if processed_data is not None:
                        # Save the DataFrame as a CSV file in the same directory
                        output_file = os.path.splitext(file_path)[0] + ".csv"
                        processed_data.to_csv(output_file, index=False)
                        print(f"Processed and saved: {output_file}")
                else:
                    print(f"Skipping non-file entry: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process all .txt files from two directories and save as CSV.")
    parser.add_argument("input_dir1", type=str, help="Path to the first input directory.")
    parser.add_argument("input_dir2", type=str, help="Path to the second input directory.")
    args = parser.parse_args()

    process_directories(args.input_dir1, args.input_dir2)
