import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CityBikesTransformer:
    def __init__(self):
        """Initialize the transformer with data quality rules"""
        self.required_columns = [
            'id', 'name', 'latitude', 'longitude', 
            'free_bikes', 'empty_slots', 'network_id', 'network_name'
        ]
        
        # Data quality thresholds
        self.quality_rules = {
            'latitude_range': (-90, 90),
            'longitude_range': (-180, 180),
            'min_station_capacity': 1,
            'max_station_capacity': 100,
            'max_name_length': 200
        }
        
        logger.info("Initialized CityBikesTransformer")
    
    def validate_data_quality(self, df):
        """
        Perform data quality checks and log issues
        Args:
            df (pandas.DataFrame): Raw data to validate
        Returns:
            dict: Data quality report
        """
        logger.info("Starting data quality validation...")
        
        quality_report = {
            'total_records': len(df),
            'issues': [],
            'duplicates': 0,
            'invalid_coordinates': 0,
            'missing_critical_fields': 0,
            'invalid_capacity_values': 0
        }
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['id']).sum()
        quality_report['duplicates'] = duplicates
        if duplicates > 0:
            quality_report['issues'].append(f"Found {duplicates} duplicate station IDs")
        
        # Check for missing critical fields
        critical_fields = ['id', 'name', 'latitude', 'longitude']
        for field in critical_fields:
            if field in df.columns:
                missing = df[field].isna().sum()
                if missing > 0:
                    quality_report['missing_critical_fields'] += missing
                    quality_report['issues'].append(f"Missing values in {field}: {missing}")
        
        # Check coordinate ranges
        if 'latitude' in df.columns and 'longitude' in df.columns:
            lat_range = self.quality_rules['latitude_range']
            lon_range = self.quality_rules['longitude_range']
            
            invalid_coords = (
                (df['latitude'].notna() & ((df['latitude'] < lat_range[0]) | (df['latitude'] > lat_range[1]))) |
                (df['longitude'].notna() & ((df['longitude'] < lon_range[0]) | (df['longitude'] > lon_range[1])))
            ).sum()
            
            quality_report['invalid_coordinates'] = invalid_coords
            if invalid_coords > 0:
                quality_report['issues'].append(f"Invalid coordinates: {invalid_coords}")
        
        # Check capacity values
        capacity_fields = ['free_bikes', 'empty_slots']
        for field in capacity_fields:
            if field in df.columns:
                invalid_capacity = (
                    (df[field].notna()) & 
                    ((df[field] < 0) | (df[field] > self.quality_rules['max_station_capacity']))
                ).sum()
                
                if invalid_capacity > 0:
                    quality_report['invalid_capacity_values'] += invalid_capacity
                    quality_report['issues'].append(f"Invalid {field} values: {invalid_capacity}")
        
        logger.info(f"Data quality validation complete. Found {len(quality_report['issues'])} issue types.")
        return quality_report
    
    def clean_station_data(self, df):
        """
        Clean and standardize station data
        Args:
            df (pandas.DataFrame): Raw station data
        Returns:
            pandas.DataFrame: Cleaned data
        """
        logger.info("Starting data cleaning...")
        df_clean = df.copy()
        
        # Remove exact duplicates
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['id'], keep='first')
        removed_duplicates = initial_count - len(df_clean)
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate stations")
        
        # Clean station names
        if 'name' in df_clean.columns:
            df_clean['name'] = df_clean['name'].astype(str)
            df_clean['name'] = df_clean['name'].str.strip()
            df_clean['name'] = df_clean['name'].str.replace(r'\s+', ' ', regex=True)  # Multiple spaces to single
            df_clean['name'] = df_clean['name'].str[:self.quality_rules['max_name_length']]  # Truncate long names
        
        # Clean numeric fields
        numeric_fields = ['latitude', 'longitude', 'free_bikes', 'empty_slots']
        for field in numeric_fields:
            if field in df_clean.columns:
                df_clean[field] = pd.to_numeric(df_clean[field], errors='coerce')
        
        # Remove records with invalid coordinates
        if 'latitude' in df_clean.columns and 'longitude' in df_clean.columns:
            lat_range = self.quality_rules['latitude_range']
            lon_range = self.quality_rules['longitude_range']
            
            valid_coords = (
                df_clean['latitude'].between(lat_range[0], lat_range[1]) &
                df_clean['longitude'].between(lon_range[0], lon_range[1])
            )
            
            invalid_count = (~valid_coords).sum()
            if invalid_count > 0:
                logger.info(f"Removing {invalid_count} records with invalid coordinates")
                df_clean = df_clean[valid_coords]
        
        # Set negative capacity values to 0 (common data issue)
        capacity_fields = ['free_bikes', 'empty_slots']
        for field in capacity_fields:
            if field in df_clean.columns:
                negative_count = (df_clean[field] < 0).sum()
                if negative_count > 0:
                    logger.info(f"Setting {negative_count} negative {field} values to 0")
                    df_clean[field] = df_clean[field].clip(lower=0)
        
        logger.info(f"Data cleaning complete. Records: {initial_count} → {len(df_clean)}")
        return df_clean
    
    def add_calculated_fields(self, df):
        """
        Add derived/calculated fields for analytics
        Args:
            df (pandas.DataFrame): Clean station data
        Returns:
            pandas.DataFrame: Data with additional fields
        """
        logger.info("Adding calculated fields...")
        df_enhanced = df.copy()
        
        # Calculate total station capacity
        if 'free_bikes' in df_enhanced.columns and 'empty_slots' in df_enhanced.columns:
            df_enhanced['total_capacity'] = (
                df_enhanced['free_bikes'].fillna(0) + df_enhanced['empty_slots'].fillna(0)
            )
        
        # Calculate utilization percentage
        if 'free_bikes' in df_enhanced.columns and 'total_capacity' in df_enhanced.columns:
            df_enhanced['utilization_pct'] = np.where(
                df_enhanced['total_capacity'] > 0,
                (df_enhanced['free_bikes'] / df_enhanced['total_capacity'] * 100).round(1),
                0
            )
        
        # Categorize station capacity
        if 'total_capacity' in df_enhanced.columns:
            df_enhanced['capacity_category'] = pd.cut(
                df_enhanced['total_capacity'],
                bins=[0, 15, 30, 50, float('inf')],
                labels=['Small', 'Medium', 'Large', 'Extra Large'],
                include_lowest=True
            )
        
        # Categorize utilization
        if 'utilization_pct' in df_enhanced.columns:
            df_enhanced['utilization_category'] = pd.cut(
                df_enhanced['utilization_pct'],
                bins=[0, 25, 50, 75, 100],
                labels=['Low', 'Medium', 'High', 'Very High'],
                include_lowest=True
            )
        
        # Add availability status
        if 'free_bikes' in df_enhanced.columns and 'empty_slots' in df_enhanced.columns:
            df_enhanced['availability_status'] = np.where(
                df_enhanced['free_bikes'] == 0, 'No Bikes Available',
                np.where(df_enhanced['empty_slots'] == 0, 'No Docks Available', 'Available')
            )
        
        # Extract location info from network location if available
        if 'network_location' in df_enhanced.columns:
            df_enhanced['city'] = df_enhanced['network_location'].str.split(',').str[0].str.strip()
            df_enhanced['country'] = df_enhanced['network_location'].str.split(',').str[-1].str.strip()
        
        # Add timestamp for when transformation occurred
        df_enhanced['transformed_at'] = datetime.now()
        
        # Create a unique key for each record
        df_enhanced['record_key'] = (
            df_enhanced['network_id'].astype(str) + '_' + 
            df_enhanced['id'].astype(str) + '_' + 
            df_enhanced['extraction_timestamp'].dt.strftime('%Y%m%d_%H%M%S')
        )
        
        logger.info(f"Added calculated fields. Final dataset has {len(df_enhanced.columns)} columns")
        return df_enhanced
    
    def standardize_data_types(self, df):
        """
        Ensure consistent data types for downstream processing
        Args:
            df (pandas.DataFrame): Enhanced data
        Returns:
            pandas.DataFrame: Data with standardized types
        """
        logger.info("Standardizing data types...")
        df_typed = df.copy()
        
        # String fields
        string_fields = ['id', 'name', 'network_id', 'network_name', 'network_company', 
                        'availability_status', 'city', 'country']
        for field in string_fields:
            if field in df_typed.columns:
                df_typed[field] = df_typed[field].astype(str)
        
        # Numeric fields
        float_fields = ['latitude', 'longitude', 'utilization_pct']
        for field in float_fields:
            if field in df_typed.columns:
                df_typed[field] = pd.to_numeric(df_typed[field], errors='coerce')
        
        int_fields = ['free_bikes', 'empty_slots', 'total_capacity']
        for field in int_fields:
            if field in df_typed.columns:
                df_typed[field] = pd.to_numeric(df_typed[field], errors='coerce').fillna(0).astype(int)
        
        # Datetime fields
        datetime_fields = ['extraction_timestamp', 'transformed_at']
        for field in datetime_fields:
            if field in df_typed.columns:
                df_typed[field] = pd.to_datetime(df_typed[field], errors='coerce')
        
        # Categorical fields
        categorical_fields = ['capacity_category', 'utilization_category']
        for field in categorical_fields:
            if field in df_typed.columns:
                df_typed[field] = df_typed[field].astype('category')
        
        logger.info("Data type standardization complete")
        return df_typed
    
    def transform(self, df):
        """
        Main transformation pipeline
        Args:
            df (pandas.DataFrame): Raw data from extractor
        Returns:
            tuple: (transformed_df, quality_report)
        """
        logger.info("Starting full transformation pipeline...")
        
        # Step 1: Data quality validation
        quality_report = self.validate_data_quality(df)
        
        # Step 2: Data cleaning
        df_clean = self.clean_station_data(df)
        
        # Step 3: Add calculated fields
        df_enhanced = self.add_calculated_fields(df_clean)
        
        # Step 4: Standardize data types
        df_final = self.standardize_data_types(df_enhanced)
        
        logger.info("Transformation pipeline complete!")
        return df_final, quality_report
    
    def save_transformed_data(self, df, network_id, output_dir="data/processed"):
        """
        Save transformed data to processed directory
        Args:
            df (pandas.DataFrame): Transformed data
            network_id (str): Network identifier
            output_dir (str): Directory to save file
        Returns:
            str: Path to saved file
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"transformed_citybikes_{network_id}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} transformed records to {filepath}")
        
        return filepath
    
    def save_quality_report(self, quality_report, network_id, output_dir="data/quality_reports"):
        """
        Save data quality report
        Args:
            quality_report (dict): Quality report from validation
            network_id (str): Network identifier
            output_dir (str): Directory to save file
        Returns:
            str: Path to saved file
        """
        import json
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"quality_report_{network_id}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Add metadata
        quality_report['generated_at'] = datetime.now().isoformat()
        quality_report['network_id'] = network_id
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        logger.info(f"Saved quality report to {filepath}")
        return filepath

# Test function
def test_transformer():
    """Test the transformer with actual data"""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.extract.citybikes_extractor import CityBikesExtractor
    
    # Step 1: Get fresh data
    print("Step 1: Extracting fresh data...")
    extractor = CityBikesExtractor()
    raw_df = extractor.extract_stations_data('citi-bike-nyc')
    
    if raw_df.empty:
        print("No data extracted. Check your internet connection.")
        return
    
    print(f"Extracted {len(raw_df)} raw records")
    print(f"Raw data columns: {list(raw_df.columns)}")
    
    # Step 2: Transform the data
    print(f"\nStep 2: Transforming data...")
    transformer = CityBikesTransformer()
    transformed_df, quality_report = transformer.transform(raw_df)
    
    print(f"Transformed data shape: {transformed_df.shape}")
    print(f"Final columns: {list(transformed_df.columns)}")
    
    # Step 3: Show quality report
    print(f"\nStep 3: Data Quality Report")
    print(f"Total records processed: {quality_report['total_records']}")
    print(f"Issues found: {len(quality_report['issues'])}")
    for issue in quality_report['issues']:
        print(f"  - {issue}")
    
    # Step 4: Show sample of transformed data
    print(f"\nStep 4: Sample of transformed data:")
    sample_cols = ['name', 'free_bikes', 'empty_slots', 'total_capacity', 
                   'utilization_pct', 'capacity_category', 'availability_status']
    available_cols = [col for col in sample_cols if col in transformed_df.columns]
    print(transformed_df[available_cols].head())
    
    # Step 5: Save the data
    print(f"\nStep 5: Saving transformed data...")
    transformed_file = transformer.save_transformed_data(transformed_df, 'citi-bike-nyc')
    quality_file = transformer.save_quality_report(quality_report, 'citi-bike-nyc')
    
    print(f"Transformed data saved to: {transformed_file}")
    print(f"Quality report saved to: {quality_file}")
    
    # Step 6: Summary statistics
    print(f"\nStep 6: Summary Statistics:")
    if 'total_capacity' in transformed_df.columns:
        print(f"Average station capacity: {transformed_df['total_capacity'].mean():.1f}")
    if 'utilization_pct' in transformed_df.columns:
        print(f"Average utilization: {transformed_df['utilization_pct'].mean():.1f}%")
    if 'availability_status' in transformed_df.columns:
        print(f"Availability status distribution:")
        print(transformed_df['availability_status'].value_counts())
    
    
  
if __name__ == "__main__":
    test_transformer()