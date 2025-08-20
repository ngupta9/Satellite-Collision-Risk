# preprocess.py
"""
Main preprocessing pipeline for satellite collision risk analysis.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import pandas as pd
from data_acquisition import TLEDataCollector
from orbital_features import OrbitFeatureExtractor
from data_utils import (save_collision_data, load_collision_data, 
                       deduplicate_satellites, prepare_training_data)
from visualization import plot_collision_probability_analysis
from config import *

def preprocess_collision_data():
    """Main preprocessing pipeline"""
    
    print("=== SATELLITE COLLISION RISK PREPROCESSING ===")
    
    # Step 1: Get TLE data
    print("\n1. Fetching TLE data from Space-Track...")
    collector = TLEDataCollector(SPACETRACK_USERNAME, SPACETRACK_PASSWORD)
    
    raw_tle_data = collector.get_active_satellites(
        limit=SATELLITE_LIMIT, 
        days_back=DAYS_BACK
    )
    tle_records = collector.parse_tle_data(raw_tle_data)
    print(f"Retrieved {len(tle_records)} satellites from last {DAYS_BACK} days")

    # Step 2: Extract orbital features
    print("\n2. Extracting orbital features...")
    extractor = OrbitFeatureExtractor()
    
    satellite_features = []
    tle_lines = {}
    
    for name, line1, line2 in tle_records:
        try:
            features = extractor.extract_orbital_elements(line1, line2)
            features['name'] = name
            satellite_features.append(features)
            tle_lines[name] = (line1, line2)
        except Exception as e:
            print(f"Error processing {name}: {e}")
    
    df = pd.DataFrame(satellite_features)
    df_unique = deduplicate_satellites(df)
    print(f"Successfully processed {len(df_unique)} unique satellites")
    
    # Step 3: Generate collision pairs (with caching)
    print("\n3. Computing collision pairs...")
    
    closest_approach_data = load_collision_data(CLOSEST_APPROACH_CACHE)
    if closest_approach_data is None:
        closest_approach_data = extractor.generate_closest_approach_data(
            df_unique, tle_lines, max_pairs=MAX_PAIRS
        )
        save_collision_data(closest_approach_data, CLOSEST_APPROACH_CACHE)
    
    print("Generating collision pairs from cached data...")
    collision_data = extractor.generate_collision_pairs_from_cache(closest_approach_data)
    
    # Step 4: Analyze and visualize
    print("\n4. Analyzing results...")
    plot_collision_probability_analysis(collision_data, save_plot=True, 
                                        output_file=PLOT_DATA_FILE)
    
    # Step 5: Prepare training data
    print("\n5. Preparing training data...")
    features, collision_probs, risk_classes = prepare_training_data(collision_data)
    
    # Save final processed data
    save_collision_data(collision_data, COLLISION_DATA_CACHE)
    
    print(f"\n=== PREPROCESSING COMPLETE ===")

    return collision_data, features, collision_probs, risk_classes

if __name__ == "__main__":
    collision_data, features, collision_probs, risk_classes = preprocess_collision_data()