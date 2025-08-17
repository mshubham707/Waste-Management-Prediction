import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from geopy.distance import geodesic
import numpy as np

def preprocess_data(data,is_file=False, output_path=None):
    """
    Preprocessing the waste management dataset: 
    rename columns, handle missing values, encode categoricals, engineer features, and scale numericals.
    
    Args:
        data (dataframe or path of raw dataset) : Path to the raw dataset (e.g., '../data/raw/train.csv') or a dataframe
        is_file (bool): Flag indicating if the input data is a file path.
        output_path (str): Path to save the processed dataset (e.g., '../data/processed/cleaned_data.csv')
    
    Returns:
        pd.DataFrame: Processed dataframe
    """
    try:
        if is_file:
            print(f"Reading data from the path {data}...")
            df = pd.read_csv(data)

        else:
            print("Reading data from the file...")
            df=pd.DataFrame(data)            

    except Exception as e:
        print(f"Error: {e}")   
        print("No input path provided, returning empty DataFrame.")
        df = pd.DataFrame()

    print("Preprocessing Started.....")
    print("Data shape:", df.shape)

    print("Initial columns:", df.columns.tolist())
    # Rename columns for consistency
    df.rename(columns={
        'City/District': 'City', 'Waste Type': 'Waste_Type', 'Waste Generated (Tons/Day)': 'Waste_Generated',
        'Recycling Rate (%)': 'Recycling_Rate', 'Population Density (People/km²)': 'Population_Density',
        'Municipal Efficiency Score (1-10)': 'Municipal_Efficiency_Score', 'Disposal Method': 'Disposal_Method',
        'Cost of Waste Management (₹/Ton)': 'Cost_of_Waste_Management', 'Awareness Campaigns Count': 'Awareness_Campaigns_Count',
        'Landfill Name': 'Landfill_Name', 'Landfill Location (Lat, Long)': 'Landfill_Location',
        'Landfill Capacity (Tons)': 'Landfill_Capacity', 'Year': 'Year'
    }, inplace=True)

    # Split Landfill_Location into Lat and Long, then drop original
    df[['Landfill_Lat', 'Landfill_Long']] = df['Landfill_Location'].str.split(',', expand=True).astype(float)
    df.drop('Landfill_Location', axis=1, inplace=True)
    df.drop('Landfill_Name', axis=1, inplace=True)  # Dropped as redundant with City

     # Feature engineering: Distance to Landfill
    city_coords = {
    'Mumbai': (19.08, 72.88),
    'Delhi': (28.65, 77.23),
    'Bengaluru': (12.97, 77.59),
    'Chennai': (13.08, 80.27),
    'Kolkata': (22.57, 88.36),
    'Hyderabad': (17.39, 78.49),
    'Pune': (18.52, 73.86),
    'Ahmedabad': (23.02, 72.57),
    'Jaipur': (26.91, 75.79),
    'Lucknow': (26.85, 80.95),
    'Surat': (21.17, 72.83),
    'Kanpur': (26.47, 80.33),
    'Nagpur': (21.15, 79.08),
    'Patna': (25.59, 85.14),
    'Bhopal': (23.26, 77.41),
    'Thiruvananthapuram': (8.52, 76.94),
    'Indore': (22.72, 75.88),
    'Vadodara': (22.31, 73.18),
    'Guwahati': (26.18, 91.75),
    'Coimbatore': (11.02, 76.96),
    'Ranchi': (23.34, 85.31),
    'Amritsar': (31.63, 74.87),
    'Jodhpur': (26.28, 73.02),
    'Varanasi': (25.32, 82.97),
    'Ludhiana': (30.90, 75.85),
    'Agra': (27.18, 78.01),
    'Meerut': (28.98, 77.71),
    'Nashik': (20.00, 73.78),
    'Rajkot': (22.30, 70.80),
    'Madurai': (9.93, 78.12),
    'Jabalpur': (23.17, 79.94),
    'Allahabad': (25.44, 81.85),
    'Visakhapatnam': (17.69, 83.22),
    'Gwalior': (26.22, 78.18),
}
    # Calculate distance to landfill for each city
    def calculate_distance(row):
        if row['City'] in city_coords:
            city_lat_long = city_coords[row['City']]
            landfill_lat_long = (row['Landfill_Lat'], row['Landfill_Long'])
            return geodesic(city_lat_long, landfill_lat_long).km
        return np.nan
    df['Distance_to_Landfill_km'] = df.apply(calculate_distance, axis=1)
  

    # Feature engineering: Years since 2019, then drop Year
    df['Years_Since_2019'] = df['Year'] - 2019
    df.drop('Year', axis=1, inplace=True)
    print(f"Final Columns: ({len(df.columns)})", df.columns.tolist())

   # Impute missing values
    num_imputer = SimpleImputer(strategy='median')
    numerical_cols = ['Waste_Generated', 'Population_Density', 'Municipal_Efficiency_Score',
                      'Cost_of_Waste_Management', 'Awareness_Campaigns_Count', 'Landfill_Capacity',
                       'Distance_to_Landfill_km', 'Years_Since_2019']
    
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    categorical_cols = ['City', 'Waste_Type', 'Disposal_Method']
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # Encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))
    df = df.drop(categorical_cols, axis=1)
    df = pd.concat([df, encoded_df], axis=1)


    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    processed_cols = numerical_cols + categorical_cols
    print(f"Processed Columns: {len(processed_cols)}", processed_cols)
    # Save processed data
    if output_path is not None:
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")

    else:
        print("No output path provided, returning processed DataFrame.")  
    print("Preprocessing completed.")  
    return df

#For testing
if __name__ == "__main__":
    print("Starting preprocessing...")
    preprocess_data(data='data/raw/train.csv', is_file=True, output_path='data/processed/cleaned_data.csv')
    print("Preprocessing completed.")