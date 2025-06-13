import requests
import pandas as pd
import yaml
import os
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CityBikesExtractor:
    def __init__(self, config_path="config/config.yaml"):
        """Initialize the extractor with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.networks_url = self.config['citybikes_api']['networks_url']
        self.base_url = self.config['citybikes_api']['base_url']
        logger.info(f"Initialized CityBikesExtractor")
    
    def get_all_networks(self):
        """
        Get list of all bike share networks
        Returns:
            pandas.DataFrame: All available networks
        """
        logger.info("Fetching all bike share networks...")
        
        try:
            response = requests.get(self.networks_url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            networks = data['networks']
            df = pd.DataFrame(networks)
            
            logger.info(f"Found {len(df)} bike share networks")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching networks: {str(e)}")
            raise
    
    def extract_network_data(self, network_id):
        """
        Extract station data for a specific network
        Args:
            network_id (str): Network identifier (e.g., 'citi-bike-nyc')
        Returns:
            dict: Network data with stations
        """
        logger.info(f"Extracting data for network: {network_id}")
        
        url = self.base_url.format(network_id=network_id)
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully extracted network data for {network_id}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching network data: {str(e)}")
            raise
    
    def extract_stations_data(self, network_id):
        """
        Extract and flatten station data for a network
        Args:
            network_id (str): Network identifier
        Returns:
            pandas.DataFrame: Flattened station data
        """
        network_data = self.extract_network_data(network_id)
        
        # Extract network info
        network_info = network_data['network']
        stations = network_info.get('stations', [])
        
        if not stations:
            logger.warning(f"No stations found for network {network_id}")
            return pd.DataFrame()
        
        # Convert stations to DataFrame
        df = pd.DataFrame(stations)
        
        # Add network metadata
        df['network_id'] = network_info.get('id')
        df['network_name'] = network_info.get('name')
        df['network_company'] = network_info.get('company', [None])[0] if network_info.get('company') else None
        df['network_location'] = f"{network_info.get('location', {}).get('city', '')}, {network_info.get('location', {}).get('country', '')}"
        df['extraction_timestamp'] = datetime.now()
        
        # Flatten nested data if present
        if 'extra' in df.columns:
            # Extract common extra fields
            extra_df = pd.json_normalize(df['extra'].fillna({}))
            df = pd.concat([df.drop('extra', axis=1), extra_df], axis=1)
        
        logger.info(f"Extracted {len(df)} stations for {network_id}")
        return df
    
    def save_to_local(self, df, network_id, output_dir="data/raw"):
        """
        Save DataFrame locally as CSV
        Args:
            df (pandas.DataFrame): Data to save
            network_id (str): Network identifier
            output_dir (str): Directory to save file
        Returns:
            str: Path to saved file
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"citybikes_{network_id}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} records to {filepath}")
        
        return filepath
    
    def get_popular_networks(self, limit=5):
        """Get a few popular networks for testing"""
        popular_networks = [
            'citi-bike-nyc',  # New York Citi Bike
            'boris-bikes',    # London Santander Cycles
            'velib',          # Paris Vélib'
            'bicing',         # Barcelona Bicing
            'divvy',          # Chicago Divvy
        ]
        return popular_networks[:limit]

# Test function
def test_extractor():
    """Test the extractor with sample data"""
    extractor = CityBikesExtractor()
    
    # Test 1: Get all networks (sample)
    print("Test 1: Getting list of all networks...")
    try:
        networks_df = extractor.get_all_networks()
        print(f"Total networks available: {len(networks_df)}")
        print(f"Sample networks:")
        print(networks_df[['id', 'name', 'location']].head(10))
    except Exception as e:
        print(f"Error getting networks: {str(e)}")
        return
    
    # Test 2: Get data for a specific network (NYC Citi Bike)
    print(f"\nTest 2: Getting station data for NYC Citi Bike...")
    
    try:
        stations_df = extractor.extract_stations_data('citi-bike-nyc')
        print(f"Stations data shape: {stations_df.shape}")
        print(f"Columns: {list(stations_df.columns)}")
        
        if len(stations_df) > 0:
            print(f"Sample station data:")
            print(stations_df[['name', 'free_bikes', 'empty_slots', 'network_name']].head())
            
            # Save locally
            filepath = extractor.save_to_local(stations_df, 'citi-bike-nyc')
            print(f"Data saved to: {filepath}")
        
    except Exception as e:
        print(f"Error in station extraction: {str(e)}")
    
    # Test 3: Show available popular networks
    print(f"\nTest 3: Popular networks for testing:")
    popular = extractor.get_popular_networks()
    for network in popular:
        print(f"  - {network}")

if __name__ == "__main__":
    test_extractor()