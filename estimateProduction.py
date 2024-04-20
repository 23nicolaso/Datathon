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
    adjusted_quarters = quarters * stage_adjustment[classification]
    
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
csv_path = 'buildings.csv'
atlanta_properties_with_estimates = load_and_classify_properties(csv_path)
print(atlanta_properties_with_estimates[['Address', 'completion_quarters']])

