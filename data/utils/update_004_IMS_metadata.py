import pandas as pd
import numpy as np
import os
import re # For natural sorting, though IMS filenames sort lexicographically

# --- Configuration for IMS Dataset ---
# This should match the 'Name' column in your metadata CSV for the IMS dataset
TARGET_DATASET_NAME_IMS = 'RM_004_IMS'  # Example name, adjust as per your metadata

# Sample rate and length from IMS dataset documentation (Qiu et al., 2006)
SAMPLE_RATE_IMS = 20000  # 20 kHz
FIXED_SAMPLE_LENTH_IMS = 20480 # Number of data points per file

# Channel configuration for IMS test sets
CHANNELS_IMS = {
    '1st_test': 8,
    '2nd_test': 4,
    '4th_test': 4  # Assuming '4th_test' corresponds to 'Set No. 3' or similar with 4 channels
}

# Degradation stage thresholds: (Normal_End_Percentage, Degradation_End_Percentage)
# Percentages are of the total number of files in the test set.
# E.g., for '1st_test', files up to 85% of total count are Normal,
# from 85% to 95% are Degradation, and beyond 95% are Fault.
BEARING_STAGE_THRESHOLDS_IMS = {
    '1st_test': (0.85, 0.95), # Normal up to 85%, Degradation up to 95%
    '2nd_test': (0.80, 0.95), # Normal up to 80%, Degradation up to 95%
    '4th_test': (0.95, 0.98)  # Normal up to 95%, Degradation up to 98%
}

# Unique labels for the 'Fault' state of each test set
# Normal = 0, Degradation = -1. Fault states will be positive integers.
IMS_SET_TYPE_TO_FAULT_STATE_LABEL = {
    '1st_test': 1,
    '2nd_test': 2,
    '4th_test': 3
}

def extract_ims_file_info(filepath_str: str):
    """
    Extracts information from the 'File' column entry for an IMS dataset.
    Assumes filepath_str is like 'set_type/filename.txt' (e.g., '1st_test/2003.10.22.12.06.24.txt').
    """
    if pd.isna(filepath_str):
        return None, None, None  # set_type, filename_str, bearing_run_id
    try:
        parts = filepath_str.split('/')
        if len(parts) >= 2: # Expecting at least 'set_type/filename'
            set_type = parts[0]
            filename_str = '/'.join(parts[1:]) # Handle filenames that might have slashes if base path included more dirs
                                                # For '1st_test/2003.10.22.12.06.24.txt', filename_str is '2003.10.22.12.06.24.txt'
            # For IMS, the 'bearing_run_id' (lifecycle being tracked) is the 'set_type' itself.
            bearing_run_id = set_type
            return set_type, filename_str, bearing_run_id
    except Exception as e:
        print(f"Error parsing IMS file path '{filepath_str}': {e}")
        return None, None, None
    print(f"Warning: Could not parse IMS file path '{filepath_str}' into expected format.")
    return None, None, None

def update_ims_metadata(
    metadata_csv_path: str,
    # data_files_root_path: str, # Not strictly needed if metadata CSV has all 'File' entries
    output_csv_path: str
) -> pd.DataFrame:
    """
    Updates the metadata CSV file with information specific to the IMS bearing dataset.
    """
    try:
        metadata_df = pd.read_csv(metadata_csv_path)
        print(f"Successfully loaded metadata, {len(metadata_df)} total records.")
    except Exception as e:
        print(f"Error reading metadata file '{metadata_csv_path}': {e}")
        return pd.DataFrame()

    # Filter for the target IMS dataset records
    ims_indices = metadata_df[metadata_df['Name'] == TARGET_DATASET_NAME_IMS].index
    if ims_indices.empty:
        print(f"No records found for dataset '{TARGET_DATASET_NAME_IMS}' in the metadata.")
        return metadata_df # Return original df if no target data
    print(f"Found {len(ims_indices)} records for '{TARGET_DATASET_NAME_IMS}' to update.")

    # --- 1. Extract initial info and prepare for file numbering ---
    temp_info_cols = ['set_type', 'filename_str', 'bearing_run_id']
    temp_df_ims = metadata_df.loc[ims_indices, 'File'].apply(
        lambda x: pd.Series(extract_ims_file_info(x), index=temp_info_cols)
    )
    for col in temp_info_cols:
        metadata_df.loc[ims_indices, col] = temp_df_ims[col]

    # --- 2. Calculate file numbers and max file numbers per set_type ---
    # This is crucial for RUL and Label calculation.
    # It assumes all files for a given set_type are present in the metadata CSV.
    file_num_map = {} # To store { (set_type, filename_str): file_num }
    max_file_num_per_set = {} # To store { set_type: max_file_num }

    valid_file_data_ims = metadata_df.loc[ims_indices].dropna(subset=['set_type', 'filename_str'])
    for set_id, group in valid_file_data_ims.groupby('set_type'):
        # IMS filenames (e.g., '2003.10.22.12.06.24.txt') sort chronologically by default
        sorted_filenames = sorted(group['filename_str'].unique())
        max_file_num_per_set[set_id] = len(sorted_filenames)
        for i, fname in enumerate(sorted_filenames):
            file_num_map[(set_id, fname)] = i + 1 # 1-indexed file number

    # Assign file_num to each row
    metadata_df.loc[ims_indices, 'file_num'] = metadata_df.loc[ims_indices].apply(
        lambda row: file_num_map.get((row['set_type'], row['filename_str']), pd.NA), axis=1
    )
    metadata_df['file_num'] = pd.to_numeric(metadata_df['file_num'], errors='coerce')


    # --- 3. Populate metadata columns for IMS dataset ---
    for index in ims_indices:
        set_type = metadata_df.loc[index, 'set_type']
        filename = metadata_df.loc[index, 'filename_str'] # Renamed for clarity from 'filename'
        current_file_num = metadata_df.loc[index, 'file_num']

        if pd.isna(set_type) or pd.isna(current_file_num):
            print(f"Skipping row index {index} due to missing set_type or file_num.")
            continue

        # Common IMS metadata
        metadata_df.loc[index, 'Sample_rate'] = SAMPLE_RATE_IMS
        metadata_df.loc[index, 'Sample_lenth'] = FIXED_SAMPLE_LENTH_IMS
        metadata_df.loc[index, 'Channel'] = CHANNELS_IMS.get(set_type, np.nan)

        # As per user request for IMS
        metadata_df.loc[index, 'Domain_id'] = np.nan
        metadata_df.loc[index, 'Domain_description'] = np.nan
        metadata_df.loc[index, 'Fault_level'] = np.nan # Or 'Fault_Severity' if that's the column

        # RUL Label calculation
        max_fn = max_file_num_per_set.get(set_type, 0)
        rul_label_val = np.nan
        if max_fn > 0 and pd.notna(current_file_num):
            if max_fn == 1:
                rul_label_val = 0.0  # Only one file, so it's the end of life
            else:
                # RUL = 1 for the first file, 0 for the last file
                rul_label_val = 1.0 - (current_file_num - 1) / (max_fn - 1)
                rul_label_val = max(0.0, min(1.0, rul_label_val)) # Ensure it's within [0,1]
        metadata_df.loc[index, 'RUL_label'] = rul_label_val

        # Label and Task applicability calculation
        current_label_val = np.nan
        fault_diag_task = True  # Default to True
        anomaly_det_task = True
        remaining_life_task = True
        digital_twin_task = True

        if max_fn > 0 and pd.notna(current_file_num) and set_type in BEARING_STAGE_THRESHOLDS_IMS:
            # Normalized position of the current file in its sequence (0 to 1)
            # current_file_normalized_index = (current_file_num -1) / (max_fn -1) if max_fn > 1 else 0
            current_file_normalized_index = current_file_num / max_fn # More direct percentage

            normal_end_perc, degradation_end_perc = BEARING_STAGE_THRESHOLDS_IMS[set_type]

            if current_file_normalized_index <= normal_end_perc: # Normal
                current_label_val = 0 # Normal state
            elif current_file_normalized_index <= degradation_end_perc: # Degradation
                current_label_val = -1 # Degradation state
                fault_diag_task = False # Fault_Diagnosis is False during degradation
            else: # Fault
                current_label_val = IMS_SET_TYPE_TO_FAULT_STATE_LABEL.get(set_type, -1000) # Specific fault label for the set_type
                # If you want a generic fault label, you could use a fixed number e.g. 1

        metadata_df.loc[index, 'Label'] = current_label_val
        metadata_df.loc[index, 'Fault_Diagnosis'] = fault_diag_task
        metadata_df.loc[index, 'Anomaly_Detection'] = anomaly_det_task
        metadata_df.loc[index, 'Remaining_Life'] = remaining_life_task
        metadata_df.loc[index, 'Digital_Twin_Prediction'] = digital_twin_task
        
        # Label_Description (optional, can be filled based on Label value)
        if 'Label_Description' in metadata_df.columns:
            if current_label_val == 0:
                metadata_df.loc[index, 'Label_Description'] = "Normal"
            elif current_label_val == -1:
                metadata_df.loc[index, 'Label_Description'] = "Degradation"
            elif pd.notna(current_label_val) and current_label_val > 0 :
                 metadata_df.loc[index, 'Label_Description'] = f"Fault_{set_type}"
            else:
                metadata_df.loc[index, 'Label_Description'] = np.nan


    # --- 4. Clean up temporary columns ---
    cols_to_drop_ims = temp_info_cols + ['file_num']
    metadata_df.drop(columns=[col for col in cols_to_drop_ims if col in metadata_df.columns], inplace=True, errors='ignore')

    # Drop Label_Description if it was added and is all NaN for IMS rows, or handle as needed.
    # For now, it's kept if the column exists.

    # --- 5. Save the updated metadata ---
    try:
        metadata_df.to_csv(output_csv_path, index=False)
        print(f"\nUpdated metadata successfully saved to: {output_csv_path}")
    except Exception as e:
        print(f"Error saving updated metadata to '{output_csv_path}': {e}")

    return metadata_df


if __name__ == '__main__':
    # --- User-defined paths ---
    # IMPORTANT: Adjust these paths to your actual file locations!
    # This is the root directory where your dataset folders (like 'a_004_IMS_raw') are located.
    # While not directly used for file listing in this version, it's good practice.
    ALL_DATASETS_ROOT_PATH = "/home/user/data/PHMbenchdata/" # e.g., /home/user/data/PHM_datasets/

    # Path to your INPUT metadata CSV file that needs updating.
    INPUT_METADATA_CSV_PATH = "/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/data/metadata_19_5_4.csv" # e.g., ./metadata_19_5_4.csv

    # Path where the UPDATED metadata CSV file will be saved.
    OUTPUT_METADATA_CSV_PATH = "/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/data/output_ims_updated_metadata.csv" # e.g., ./RM_004_IMS_updated_metadata.csv

    print(f"--- IMS Dataset Metadata Update Script ---")
    print(f"All datasets root path (for reference): {ALL_DATASETS_ROOT_PATH}")
    print(f"Input metadata CSV: {INPUT_METADATA_CSV_PATH}")
    print(f"Output metadata CSV: {OUTPUT_METADATA_CSV_PATH}")

    # Basic path checks (optional, but good for user feedback)
    if not os.path.exists(INPUT_METADATA_CSV_PATH):
        print(f"ERROR: Input metadata file not found at '{INPUT_METADATA_CSV_PATH}'")
        print("Please ensure the path is correct.")
    # You might also check for the existence of the specific dataset folder if needed:
    # ims_data_path_check = os.path.join(ALL_DATASETS_ROOT_PATH, TARGET_DATASET_NAME_IMS)
    # if not os.path.exists(ims_data_path_check):
    #     print(f"WARNING: IMS dataset folder not found at '{ims_data_path_check}' (for reference).")
    else:
        print(f"\nStarting update for IMS dataset ('{TARGET_DATASET_NAME_IMS}')...")
        updated_df = update_ims_metadata(
            metadata_csv_path=INPUT_METADATA_CSV_PATH,
            # data_files_root_path=ALL_DATASETS_ROOT_PATH, # Pass if function uses it
            output_csv_path=OUTPUT_METADATA_CSV_PATH
        )

        if not updated_df.empty:
            print("\n--- Overview of updated IMS data in the dataframe ---")
            # Display a sample of the updated rows for verification
            ims_subset_display = updated_df[updated_df['Name'] == TARGET_DATASET_NAME_IMS].copy()
            if not ims_subset_display.empty:
                # Add back temporary columns for display if needed for understanding
                # temp_display_info = ims_subset_display['File'].apply(lambda x: pd.Series(extract_ims_file_info(x), index=['set_type_disp', 'filename_disp', 'run_id_disp']))
                # ims_subset_display['set_type_disp'] = temp_display_info['set_type_disp']
                # ims_subset_display['filename_disp'] = temp_display_info['filename_disp']


                columns_to_show = [
                    'File', 'Label','Label_Description', 'RUL_label', 'Sample_rate', 'Sample_lenth', 'Channel',
                    'Fault_Diagnosis', 'Anomaly_Detection', 'Remaining_Life', 'Digital_Twin_Prediction',
                    'Domain_id' # 'set_type_disp', 'filename_disp' # if added
                ]
                columns_to_show = [col for col in columns_to_show if col in ims_subset_display.columns]

                print(f"\nSample of updated '{TARGET_DATASET_NAME_IMS}' records (first 5 and last 5):")
                if len(ims_subset_display) > 10:
                    print(pd.concat([ims_subset_display.head(5), ims_subset_display.tail(5)])[columns_to_show])
                else:
                    print(ims_subset_display[columns_to_show])

                print(f"\nValue counts for 'Label' in '{TARGET_DATASET_NAME_IMS}':")
                print(ims_subset_display['Label'].value_counts(dropna=False).sort_index())

                print(f"\nValue counts for 'Channel' in '{TARGET_DATASET_NAME_IMS}':")
                print(ims_subset_display['Channel'].value_counts(dropna=False).sort_index())

                # Check RUL_label range
                if 'RUL_label' in ims_subset_display.columns:
                     print(f"\nRUL_label statistics for '{TARGET_DATASET_NAME_IMS}':")
                     print(ims_subset_display['RUL_label'].agg(['min', 'max', 'mean', 'count']))


            else:
                print(f"No data for '{TARGET_DATASET_NAME_IMS}' found in the returned dataframe for display.")
    print("\nScript execution finished.")