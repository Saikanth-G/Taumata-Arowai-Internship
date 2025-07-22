import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import os

# --- Configuration ---
# Define paths to your internal data and where the final output will be saved
# Since all files are in the same folder, we just use the filenames directly.
INTERNAL_DATA_PATH = 'internal dataset.xlsx' # Updated to .xlsx
OUTPUT_EXCEL_PATH = 'final_combined_water_data.xlsx' # A clear new output name
DISTANCE_THRESHOLD_KM = 0.5  # 0.5 kilometers = 500 meters

# List of LAWA datasets and their sheet names, along with column mapping for standardization
# File paths are now just the filenames, as they are in the same directory as the script.
LAWA_DATASETS_INFO = [
    {
        'file_path': 'Estuary Health dataset.xlsx',
        'sheet_name': 'Data',
        'dataset_type': 'Estuary',
        'column_mapping': {
            'LAWA Site ID': 'LAWA_Site_ID',
            'Date sampled (YYYY/MM/DD)': 'Sample_Date_Time',
            'Indicator': 'Indicator_Name',
            'Indicator value': 'Indicator_Value',
            'Units': 'Units',
            'Standard deviation': 'Standard_Deviation',
            'Council/Agency': 'Agency'
        }
    },
    {
        'file_path': 'Estuary Health dataset.xlsx',
        'sheet_name': 'Site list',
        'dataset_type': 'Estuary_Site_Metadata',
        'column_mapping': {
            'LAWA Site ID': 'LAWA_Site_ID',
            'Estuary ID': 'Estuary_ID',
            'Estuary name': 'Estuary_Name',
            'Estuary type': 'Estuary_Type',
            'Intertidal/Subtidal': 'Intertidal_Subtidal',
            'Monitored indicators': 'Monitored_Indicators_Flag',
            'Council/Agency': 'Agency'
        }
    },
    {
        'file_path': 'Groundwater quality dataset (2004-2022).xlsx',
        'sheet_name': 'GWQMonitoringResults2004-22',
        'dataset_type': 'Groundwater_Monitoring',
        'column_mapping': {
            'LAWASiteID': 'LAWA_Site_ID',
            'LAWAWellName': 'LAWA_Well_Name',
            'Date': 'Sample_Date_Time',
            'Indicator': 'Indicator_Name',
            'RawValue': 'Raw_Value',
            'Value': 'Value_Cleaned',
            'Units': 'Units'
        }
    },
    {
        'file_path': 'Groundwater quality dataset (2004-2022).xlsx',
        'sheet_name': 'State Results',
        'dataset_type': 'Groundwater_State',
        'column_mapping': {
            'LAWASiteID': 'LAWA_Site_ID',
            'LAWAWellName': 'LAWA_Well_Name',
            'Indicator': 'Indicator_Name',
            'State': 'State_Result',
            'Units': 'Units'
        }
    },
    {
        'file_path': 'Groundwater quality dataset (2004-2022).xlsx',
        'sheet_name': 'Trend Results',
        'dataset_type': 'Groundwater_Trend',
        'column_mapping': {
            'LAWASiteID': 'LAWA_Site_ID',
            'LAWAWellName': 'LAWA_Well_Name',
            'Indicator': 'Indicator_Name',
            'TrendPeriod (years)': 'Trend_Period_Years',
            'TrendDescription': 'Trend_Description'
        }
    },
    {
        'file_path': 'Lake water quality dataset (2004-2023).xlsx',
        'sheet_name': 'Monitoring Dataset (2004-2023)',
        'dataset_type': 'Lake_Monitoring',
        'column_mapping': {
            'LawaSiteID': 'LAWA_Site_ID',
            'LFENZID': 'FENZ_ID',
            'SampleDateTime': 'Sample_Date_Time',
            'Indicator': 'Indicator_Name',
            'Value': 'Indicator_Value',
            'Value (Agency)': 'Value_Agency_Reported',
            'Units': 'Units',
            'GeomorphicLType': 'Geomorphic_Lake_Type',
            'LTypeMixingPattern': 'Lake_Mixing_Pattern'
        }
    },
    {
        'file_path': 'Lake water quality dataset (2004-2023).xlsx',
        'sheet_name': 'State',
        'dataset_type': 'Lake_State',
        'column_mapping': {
            'LawaSiteID': 'LAWA_Site_ID',
            'LFENZID': 'FENZ_ID',
            'hYear': 'Hydro_Year',
            'Indicator / Attribute': 'Indicator_Name',
            'AttributeBand': 'Attribute_Band',
            'Median': 'Median_Value',
            'Maximum': 'Maximum_Value',
            '95thPercentile': '95th_Percentile_Value',
            'Units': 'Units',
            'GeomorphicLType': 'Geomorphic_Lake_Type',
            'LakeTypeMixing': 'Lake_Mixing_Pattern'
        }
    },
    {
        'file_path': 'Lake water quality dataset (2004-2023).xlsx',
        'sheet_name': 'Trend',
        'dataset_type': 'Lake_Trend',
        'column_mapping': {
            'LawaSiteID': 'LAWA_Site_ID',
            'LFENZID': 'FENZ_ID',
            'Indicator': 'Indicator_Name',
            'TrendPeriod (years)': 'Trend_Period_Years',
            'Trend Score': 'Trend_Score',
            'Trend Description': 'Trend_Description'
        }
    },
    {
        'file_path': 'Lake water quality dataset (2004-2023).xlsx',
        'sheet_name': 'TLI',
        'dataset_type': 'Lake_TLI',
        'column_mapping': {
            'LawaSiteID': 'LAWA_Site_ID',
            'FENZID': 'FENZ_ID',
            'Lake': 'Lake_Name',
            'TLIhYear': 'TLI_Hydro_Year',
            'TLI': 'TLI_Score',
            'MixingPattern': 'Lake_Mixing_Pattern',
            'GeomorphicType': 'Geomorphic_Lake_Type'
        }
    },
    {
        'file_path': 'River macroinvertebrate monitoring results (2004-2023).xlsx',
        'sheet_name': 'Monitoring results (2004-2023)',
        'dataset_type': 'River_Macro_Monitoring',
        'column_mapping': {
            'LawaSiteID': 'LAWA_Site_ID',
            'SampleDateTime': 'Sample_Date_Time',
            'Indicator': 'Indicator_Name',
            'Value': 'Indicator_Value',
            'RECLandCover': 'REC_Land_Cover'
        }
    },
    {
        'file_path': 'River macroinvertebrate state and trend results (Sept 2024).xlsx',
        'sheet_name': 'RiverEcology State',
        'dataset_type': 'River_Macro_State',
        'column_mapping': {
            'LawaSiteID': 'LAWA_Site_ID',
            'Year': 'Reporting_Year',
            'Indicator': 'Indicator_Name',
            'Median': 'Median_Value',
            'Attribute Band': 'Attribute_Band',
            'RECLandCover': 'REC_Land_Cover'
        }
    },
    {
        'file_path': 'River macroinvertebrate state and trend results (Sept 2024).xlsx',
        'sheet_name': 'RiverEcology Trend',
        'dataset_type': 'River_Macro_Trend',
        'column_mapping': {
            'LawaSiteID': 'LAWA_Site_ID',
            'Indicator': 'Indicator_Name',
            'TrendPeriod (year)': 'Trend_Period_Years',
            'Trend Score': 'Trend_Score',
            'TrendDescription': 'Trend_Description',
            'RECLandCover': 'REC_Land_Cover'
        }
    },
    {
        'file_path': 'River water quality state and trend results (Sept 2024).xlsx',
        'sheet_name': 'State Quartile',
        'dataset_type': 'River_WQ_State_Quartile',
        'column_mapping': {
            'LawaSiteID': 'LAWA_Site_ID',
            'Indicator': 'Indicator_Name',
            'Median_AllSites': 'Median_All_Sites',
            'quartile_AllSites': 'Quartile_All_Sites',
            'quartile__SameLandUse': 'Quartile_Same_LandUse',
            'quartile_SameAltitude': 'Quartile_Same_Altitude',
            'quartile_SameAlt_SameLU': 'Quartile_Same_Alt_Same_LU',
            'WFSLanduse': 'WFS_Land_Use',
            'WFSAltitude': 'WFS_Altitude',
            'RECLandCover': 'REC_Land_Cover'
        }
    },
    {
        'file_path': 'River water quality state and trend results (Sept 2024).xlsx',
        'sheet_name': 'State Attribute Band',
        'dataset_type': 'River_WQ_State_Attribute',
        'column_mapping': {
            'LAWASiteID': 'LAWA_Site_ID',
            'hYear': 'Hydro_Year',
            'Indicator / Attribute': 'Indicator_Name',
            'Attribute Band': 'Attribute_Band',
            'MedianValue': 'Median_Value',
            '95th Percentile': '95th_Percentile_Value',
            'WFSLandUse': 'WFS_Land_Use',
            'WFSAltitude': 'WFS_Altitude',
            'RECLandCover': 'REC_Land_Cover'
        }
    },
    {
        'file_path': 'River water quality state and trend results (Sept 2024).xlsx',
        'sheet_name': 'Trend',
        'dataset_type': 'River_WQ_Trend',
        'column_mapping': {
            'LawaSiteID': 'LAWA_Site_ID',
            'Indicator': 'Indicator_Name',
            'TrendPeriod': 'Trend_Period_Years',
            'TrendDataFrequency': 'Trend_Data_Frequency',
            'TrendScore': 'Trend_Score',
            'TrendDescription': 'Trend_Description',
            'WFSLandUse': 'WFS_Land_Use',
            'WFSAltitude': 'WFS_Altitude',
            'RECLandCover': 'REC_Land_Cover'
        }
    }
]

# --- Helper Functions (No changes needed here) ---

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on Earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    R = 6371  # Earth radius in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def load_and_standardize_lawa_data(file_info):
    """
    Loads data from an Excel sheet, adds a source_dataset column,
    renames columns, and ensures correct data types for Lat/Lon and dates.
    """
    file_path = file_info['file_path']
    sheet_name = file_info['sheet_name']
    dataset_type = file_info['dataset_type']
    column_mapping = file_info['column_mapping']

    print(f"Processing: {file_path} - {sheet_name}")

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"Error: LAWA data file not found at {file_path}. Please check the filename and ensure it's in the same folder as the script. Skipping.")
        return pd.DataFrame() # Return empty DataFrame if file not found
    except Exception as e:
        print(f"Error reading {file_path} - {sheet_name}: {e}. Skipping.")
        return pd.DataFrame()

    df['source_dataset'] = f"{dataset_type} - {sheet_name}"

    # Rename columns
    df.rename(columns=column_mapping, inplace=True)

    # Ensure Latitude and Longitude are float
    for col in ['Latitude', 'Longitude']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert common date/time columns to datetime objects
    date_cols_to_convert = ['Sample_Date_Time', 'DateImported', 'Date'] # Added 'Date' for groundwater
    for col in date_cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df

# --- Main Execution (Minor changes to internal data loading) ---

def main():
    print("Starting data integration process...")

    # A. Load Internal Data
    try:
        # Changed from pd.read_csv to pd.read_excel
        internal_df = pd.read_excel(INTERNAL_DATA_PATH)
        internal_df.rename(columns={
            'Latitude': 'Internal_Latitude',
            'Longitude': 'Internal_Longitude',
            'Supply UUID': 'Supply_UUID'
        }, inplace=True)
        # Ensure UUID is string and Lat/Lon are float
        internal_df['Supply_UUID'] = internal_df['Supply_UUID'].astype(str)
        internal_df['Internal_Latitude'] = pd.to_numeric(internal_df['Internal_Latitude'], errors='coerce')
        internal_df['Internal_Longitude'] = pd.to_numeric(internal_df['Internal_Longitude'], errors='coerce')
        # Drop rows with NaN in critical columns
        internal_df.dropna(subset=['Supply_UUID', 'Internal_Latitude', 'Internal_Longitude'], inplace=True)
        print(f"Loaded internal data: {len(internal_df)} rows.")
    except FileNotFoundError:
        print(f"Error: Internal data file not found at {INTERNAL_DATA_PATH}. Please ensure 'internal dataset.xlsx' is in the same folder as this script. Exiting.")
        return
    except Exception as e:
        print(f"Error loading internal data: {e}. Exiting.")
        return

    # B. Load and Combine LAWA Data
    print("\nLoading and standardizing LAWA datasets...")
    lawa_dfs = []
    for info in LAWA_DATASETS_INFO:
        df = load_and_standardize_lawa_data(info)
        if not df.empty:
            lawa_dfs.append(df)

    if not lawa_dfs:
        print("No LAWA data loaded. Exiting.")
        return

    combined_lawa_df = pd.concat(lawa_dfs, ignore_index=True)

    # Drop rows from combined_lawa_df where Latitude or Longitude are NaN, as these are critical for matching
    combined_lawa_df.dropna(subset=['Latitude', 'Longitude'], inplace=True, how='any') # 'how=any' means drop if either is NaN

    # Optional: Remove exact duplicates from the combined LAWA data if necessary
    unique_cols_for_lawa = [
        'LAWA_Site_ID', 'Sample_Date_Time', 'Indicator_Name', 'Indicator_Value',
        'Latitude', 'Longitude', 'source_dataset', 'Units' # Added Units to enhance uniqueness for measurements
    ]
    # Filter to only existing columns for the drop_duplicates subset
    unique_cols_for_lawa_existing = [col for col in unique_cols_for_lawa if col in combined_lawa_df.columns]
    
    initial_lawa_rows = len(combined_lawa_df)
    combined_lawa_df.drop_duplicates(subset=unique_cols_for_lawa_existing, keep='first', inplace=True)
    print(f"Combined LAWA data: {initial_lawa_rows} rows before de-duplication, {len(combined_lawa_df)} rows after de-duplication.")


    # C. Perform Spatial Matching
    print(f"\nPerforming spatial matching with a threshold of {DISTANCE_THRESHOLD_KM} km...")
    matched_records = []
    matched_internal_uuids = set()

    # Iterate through internal data points
    for idx_internal, row_internal in internal_df.iterrows():
        internal_lat = row_internal['Internal_Latitude']
        internal_lon = row_internal['Internal_Longitude']
        supply_uuid = row_internal['Supply_UUID']

        # Skip if internal lat/lon are NaN (already dropped in previous step, but good defensive check)
        if pd.isna(internal_lat) or pd.isna(internal_lon):
            continue

        # Calculate distances to all LAWA points
        distances = combined_lawa_df.apply(
            lambda x: haversine_distance(internal_lat, internal_lon, x['Latitude'], x['Longitude']),
            axis=1
        )

        # Find potential matches within the threshold
        potential_matches = combined_lawa_df[distances <= DISTANCE_THRESHOLD_KM].copy()

        if not potential_matches.empty:
            matched_internal_uuids.add(supply_uuid)
            potential_matches['Match_Distance_km'] = distances[potential_matches.index]

            # Add internal data columns to each matched external row
            # Use .loc for setting values on a slice to avoid SettingWithCopyWarning
            for col in internal_df.columns:
                potential_matches.loc[:, col] = row_internal[col]
            
            matched_records.append(potential_matches)

    # D. Add Non-Matched Internal UUIDs
    unmatched_internal_uuids = set(internal_df['Supply_UUID']) - matched_internal_uuids
    unmatched_internal_df = internal_df[internal_df['Supply_UUID'].isin(unmatched_internal_uuids)].copy()

    if not unmatched_internal_df.empty:
        unmatched_internal_df['source_dataset'] = 'No Match'
        unmatched_internal_df['Match_Distance_km'] = np.nan
        unmatched_internal_df['LAWA_Site_ID'] = 'No Match' # Placeholder for unmatched
        
        # To ensure consistent columns when concatenating unmatched rows
        # We collect all unique columns found in the LAWA dataframes
        all_lawa_columns = set()
        for df_lawa in lawa_dfs: # Use the original list of individual LAWA DFs
             all_lawa_columns.update(df_lawa.columns)
        
        # Add internal and matching specific columns
        all_lawa_columns.update(['Supply_UUID', 'Internal_Latitude', 'Internal_Longitude', 'Match_Distance_km'])

        # Create a dictionary to hold the data for the unmatched rows, with NaNs for LAWA-specific columns
        unmatched_data_dict = {}
        for col in all_lawa_columns:
            if col in unmatched_internal_df.columns:
                unmatched_data_dict[col] = unmatched_internal_df[col].tolist()
            else:
                unmatched_data_dict[col] = [np.nan] * len(unmatched_internal_df)
        
        unmatched_df_final = pd.DataFrame(unmatched_data_dict)
        
        if not unmatched_df_final.empty:
            matched_records.append(unmatched_df_final)
        print(f"Found {len(unmatched_internal_uuids)} internal UUIDs with no matches.")


    # E. Create Final DataFrame
    if not matched_records:
        print("No matches found and no unmatched internal data to report. Output file will not be created.")
        return

    final_df = pd.concat(matched_records, ignore_index=True)

    # Define a desired column order for better readability
    desired_order = [
        'Supply_UUID',
        'Internal_Latitude',
        'Internal_Longitude',
        'Match_Distance_km',
        'source_dataset',
        'LAWA_Site_ID',
        'FENZ_ID', # FENZ ID is critical for Lake data
        'Latitude', # External LAWA Lat/Lon
        'Longitude', # External LAWA Lat/Lon
        'Region',
        'Agency',
        'SiteID',
        'Site name',
        'LAWA_Well_Name',
        'Estuary_Name',
        'Estuary_ID',
        'Estuary_Type',
        'Intertidal_Subtidal',
        'Monitored_Indicators_Flag',
        # Specific indicator flags from Estuary Site List (if they appear in concatenated data)
        'Mud Content', 'Lead (Pb)', 'Copper (Cu)', 'Zinc (Zn)', 'Cadmium (Cd)', 'Chromium (Cr)',
        'Nickel (Ni)', 'Silver (Ag)', 'Mercury (Hg)', 'Arsenic (As)', 'Total PAH', 'Total DDT', 'Estuary macrofauna score',
        'CouncilSiteID',
        'Geomorphic_Lake_Type',
        'Lake_Mixing_Pattern',
        'Lake_Name',
        'RECLandCover', # Original name if not mapped
        'REC_Land_Cover', # Standardized name
        'Catchment',
        'WFSLanduse', # Original name if not mapped
        'WFS_Land_Use', # Standardized name
        'WFSAltitude', # Original name if not mapped
        'WFS_Altitude', # Standardized name
        'SedimentClass',
        'Indicator_Name', # Key measurement identifier
        'Sample_Date_Time', # Date/Time of specific measurement
        # All "value" type columns - keep their distinct names
        'Indicator_Value',
        'Raw_Value',
        'Value_Cleaned',
        'State_Result',
        'Median_Value',
        'Maximum_Value',
        '95th_Percentile_Value',
        'TLI_Score',
        'Units',
        'Attribute_Band',
        'Trend_Period_Years',
        'Trend_Description',
        'Trend_Score',
        'Hydro_Year',
        'Reporting_Year',
        'TLI_Hydro_Year',
        'Timezone',
        'Value_Agency_Reported',
        'Symbol',
        'QC (Agency)',
        'QCNEMSEquivalent',
        'Agency Indicator Name',
        'CenType',
        'Trend_Data_Frequency',
        'Median_All_Sites',
        'Quartile_All_Sites',
        'Quartile_Same_LandUse',
        'Quartile_Same_Altitude',
        'Quartile_Same_Alt_Same_LU',
        'number_of_data_points_CHLA', 'number_of_data_points_Secchi',
        'number_of_data_points_TN', 'number_of_data_points_TP',
        'overall_data_freq'
    ]

    # Filter to only columns that actually exist in the final_df
    final_cols = [col for col in desired_order if col in final_df.columns]
    # Add any columns not in desired_order but present in final_df (e.g., if a new column appears)
    for col in final_df.columns:
        if col not in final_cols:
            final_cols.append(col)

    final_df = final_df[final_cols]

    # F. Save to Excel
    try:
        final_df.to_excel(OUTPUT_EXCEL_PATH, index=False)
        print(f"\nData successfully combined and saved to '{OUTPUT_EXCEL_PATH}'")
        print(f"Total rows in final dataset: {len(final_df)}")
    except Exception as e:
        print(f"Error saving data to Excel: {e}")

if __name__ == "__main__":
    # Create the output directory if it doesn't exist (though not strictly needed if output is in same dir)
    output_dir = os.path.dirname(OUTPUT_EXCEL_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main()