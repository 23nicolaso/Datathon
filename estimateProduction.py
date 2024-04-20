import pandas as pd
import os   

def estimate_completion_quarters(size_sf, classification):
    # Estimate the number of quarters to complete based on size
    if size_sf < 100000:
        quarters = 2
    elif 100000 <= size_sf < 300000:
        quarters = 3
    elif 300000 <= size_sf < 600000:
        quarters = 4
    elif 600000 <= size_sf < 1000000:
        quarters = 5
    else:
        quarters = 6
    
    # Adjust the estimate based on the classification stage
    stage_adjustment = {'U': 1, 'G': 0.75, 'C': 0.5, 'F': 0.25, 'N': 0}
    adjusted_quarters = round(quarters * stage_adjustment[classification])
    
    return adjusted_quarters

def load_and_classify_properties(csv_path):
    # Load the CSV file
    properties_df = pd.read_csv(csv_path)
    
    # Filter for Atlanta properties
    atlanta_properties = properties_df[properties_df['MarketCode'] == 'ATLANT']

    atlanta_properties['completion_quarters'] = atlanta_properties.apply(
        lambda row: estimate_completion_quarters(row['Size_sf'], row['AnsType']),
        axis=1
    )
    
    return atlanta_properties

# Example usage
csv_path = 'trainingData.csv'
atlanta_properties_with_estimates = load_and_classify_properties(csv_path)
print(atlanta_properties_with_estimates[['Address', 'completion_quarters']])

def total_square_feet_per_quarter(properties_df):
    """
    Calculate the total number of square feet to be completed each coming quarter.
    
    Args:
    properties_df (DataFrame): DataFrame containing property information including 'Size_sf' and 'completion_quarters'.
    
    Returns:
    DataFrame: A DataFrame with quarters as index and total square feet to be completed in each quarter.
    """
    # Group by 'completion_quarters' and sum 'Size_sf' for each group
    total_sf_per_quarter = properties_df.groupby('completion_quarters')['Size_sf'].sum().reset_index()
    
    # Rename columns for clarity
    total_sf_per_quarter.columns = ['Quarter', 'Total Square Feet']
    
    # Set 'Quarter' as index
    total_sf_per_quarter.set_index('Quarter', inplace=True)
    
    return total_sf_per_quarter

# Example usage
total_sf_per_quarter = total_square_feet_per_quarter(atlanta_properties_with_estimates)
print(total_sf_per_quarter)

def save_estimated_quarters_to_csv(properties_df, output_csv_path):
    """
    Save the DataFrame containing properties and their estimated quarters until completion into a CSV file.
    
    Args:
    properties_df (DataFrame): DataFrame containing property information including estimated quarters until completion.
    output_csv_path (str): Path to the output CSV file.
    """
    # Select relevant columns to save
    output_df = properties_df[['Address', 'completion_quarters']]
    
    # Save to CSV
    output_df.to_csv(output_csv_path, index=False)
    print(f"Saved estimated quarters until completion to {output_csv_path}")

# Example usage
output_csv_path = 'estimated_quarters_until_completion.csv'
save_estimated_quarters_to_csv(atlanta_properties_with_estimates, output_csv_path)
