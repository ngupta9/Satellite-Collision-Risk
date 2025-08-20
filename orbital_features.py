'''
orbital_features.py
This module extracts orbital elements from TLE (Two-Line Element) data, 
computes closest approach metrics between satellite pairs, calculates 
multi-factor collision probabilities, and generates collision datasets 
with features and risk classifications for machine learning.
'''
from skyfield.api import load, EarthSatellite
import numpy as np
import math

class OrbitFeatureExtractor:
    """
    This class extracts orbital features from TLE data.
    """
    def __init__(self):
        """
        Initialize the OrbitFeatureExtractor.
        """
        self.ts = load.timescale()
    
    def extract_orbital_elements(self, tle_line1, tle_line2):
        """
        Extract orbital elements from TLE
        Args:
            tle_line1 (str): The first line of the TLE.
            tle_line2 (str): The second line of the TLE.
        Returns:
            dict: A dictionary containing the extracted orbital elements.
        """
        sat = EarthSatellite(tle_line1, tle_line2, ts=self.ts)
        
        # Get current position and velocity
        t = self.ts.now()
        geocentric = sat.at(t)
        
        return {
            'semi_major_axis': sat.model.a,
            'eccentricity': sat.model.ecco, 
            'inclination': np.degrees(sat.model.inclo),
            'raan': np.degrees(sat.model.nodeo),  # Right ascension
            'arg_perigee': np.degrees(sat.model.argpo),
            'mean_anomaly': np.degrees(sat.model.mo),
            'mean_motion': sat.model.no_kozai * 24,  # revs per day
            'altitude': geocentric.distance().km - 6371,  # km above Earth
        }
    
    def compute_closest_approach(self, tle_line1_sat1, tle_line2_sat1, 
                           tle_line1_sat2, tle_line2_sat2, 
                           hours_ahead=24, time_step_minutes=10):
        """
        Compute the closest approach between two satellites over a time period.
        Args:
            tle_line1_sat1, tle_line2_sat1: TLE lines for satellite 1
            tle_line1_sat2, tle_line2_sat2: TLE lines for satellite 2
            hours_ahead: Time period to check (hours)
            time_step_minutes: Time resolution (minutes)
        Returns:
            dict: Contains closest_approach_km, time_of_closest_approach, relative_velocity
        """
        # Create satellite objects
        sat1 = EarthSatellite(tle_line1_sat1, tle_line2_sat1, ts=self.ts)
        sat2 = EarthSatellite(tle_line1_sat2, tle_line2_sat2, ts=self.ts)
        
        # Generate time array
        t0 = self.ts.now()
        times = []
        for i in range(0, int(hours_ahead * 60), time_step_minutes):
            times.append(t0 + (i / (24 * 60)))  # Add minutes as fraction of day
        
        # Compute positions at each time
        distances = []
        relative_velocities = []
        
        for t in times:
            # Get geocentric positions
            pos1 = sat1.at(t)
            pos2 = sat2.at(t)
            
            # Compute distance between satellites
            diff = pos1.position.km - pos2.position.km
            distance = np.linalg.norm(diff)
            distances.append(distance)
            
            # Compute relative velocity (optional - useful for collision assessment)
            vel1 = pos1.velocity.km_per_s
            vel2 = pos2.velocity.km_per_s
            rel_vel = np.linalg.norm(vel1 - vel2)
            relative_velocities.append(rel_vel)
        
        # Find closest approach
        min_distance_idx = np.argmin(distances)
        closest_approach_km = distances[min_distance_idx]
        time_of_closest_approach = times[min_distance_idx]
        relative_velocity_at_closest = relative_velocities[min_distance_idx]
        
        return {
            'closest_approach_km': closest_approach_km,
            'time_of_closest_approach': time_of_closest_approach,
            'relative_velocity_kms': relative_velocity_at_closest,
            'all_distances': distances  # For debugging/visualization
        }
    
    def compute_collision_probability(self, closest_approach_km, relative_velocity_kms, 
                                        altitude_diff, inclination_diff, 
                                        sat1_eccentricity, sat2_eccentricity):
        """
        Multi-factor collision probability using orbital mechanics principles.
        Args:
            closest_approach_km (float): Closest approach distance in kilometers.
            relative_velocity_kms (float): Relative velocity at closest approach in km/s.
            altitude_diff (float): Altitude difference between satellites in kilometers.
            inclination_diff (float): Inclination difference between satellites in degrees.
            sat1_eccentricity (float): Eccentricity of satellite 1.
            sat2_eccentricity (float): Eccentricity of satellite 2.
        Returns:
            float: Collision probability (0 to 1).
        """
        
        # Base probability from distance (exponential decay)
        base_prob = 5e-4 * math.exp(-closest_approach_km / 80)
        
        # Velocity factor: Higher relative velocity = higher risk
        # Typical orbital velocities: 7-15 km/s
        velocity_factor = 1.0 + (relative_velocity_kms - 10.0) / 15.0
        velocity_factor = max(0.3, min(velocity_factor, 3.0))  # Clamp 0.3-3.0x

        # Altitude similarity: Similar altitudes = higher interaction probability
        # Objects in different orbital "shells" interact less
        altitude_factor = math.exp(-altitude_diff / 2000.0)  # Decay over 2000km
        altitude_factor = max(0.1, altitude_factor)  # Minimum 0.1x
        
        # Inclination similarity: Similar orbital planes = more frequent crossings
        inclination_factor = math.exp(-inclination_diff / 45.0)  # Decay over 45 degrees
        inclination_factor = max(0.2, inclination_factor)  # Minimum 0.2x
        
        # Eccentricity factor: High eccentricity = harder to predict = slightly higher risk
        avg_eccentricity = (sat1_eccentricity + sat2_eccentricity) / 2.0
        eccentricity_factor = 1.0 + avg_eccentricity * 2.0  # Up to 2x for very eccentric
        eccentricity_factor = min(eccentricity_factor, 2.0)
        
        # Combined probability
        total_prob = (base_prob * 
                    velocity_factor * 
                    altitude_factor * 
                    inclination_factor * 
                    eccentricity_factor)
        
        return max(total_prob, 1e-9)
    
    def generate_closest_approach_data(self, df, tle_lines, max_pairs=5000):
        """
        Compute and cache the expensive orbital mechanics calculations
        Args:
            df (pd.DataFrame): DataFrame containing satellite features.
            tle_lines (dict): Dictionary mapping satellite names to their TLE lines.
            max_pairs (int): Maximum number of satellite pairs to process.
        Returns:
            closest_approach_data (list): List of dictionaries containing closest approach data for each pair.
        """
        import itertools
        import random
        
        satellite_data = df.to_dict('records')
        all_pairs = list(itertools.combinations(satellite_data, 2))
        random.shuffle(all_pairs)
        
        closest_approach_data = []
        
        print(f"Computing closest approaches for {min(max_pairs, len(all_pairs))} pairs...")
        
        for i, (sat1, sat2) in enumerate(all_pairs[:max_pairs]):
            if i % 500 == 0:
                print(f"Processing pair {i}/{max_pairs}")
                
            try:
                sat1_tle = tle_lines[sat1['name']]
                sat2_tle = tle_lines[sat2['name']]
                
                # Only do the expensive orbital mechanics calculation
                closest_approach_result = self.compute_closest_approach(
                    sat1_tle[0], sat1_tle[1], sat2_tle[0], sat2_tle[1]
                )
                
                # Store all the raw orbital data
                pair_data = {
                    'sat1_id': sat1['name'],
                    'sat2_id': sat2['name'],
                    'sat1_elements': sat1,
                    'sat2_elements': sat2,
                    'closest_approach_km': closest_approach_result['closest_approach_km'],
                    'relative_velocity_kms': closest_approach_result['relative_velocity_kms'],
                }
                
                closest_approach_data.append(pair_data)
                
            except Exception as e:
                print(f"Error processing pair {sat1['name']}-{sat2['name']}: {e}")
        
        return closest_approach_data

    def generate_collision_pairs_from_cache(self, closest_approach_data):
        """
        Generate collision pairs using multi-factor collision probability.
        Args:
            closest_approach_data (list): List of dictionaries containing closest approach data.
        Returns:
            collision_data (list): List of dictionaries containing collision pair data.
        """
        
        collision_data = []
        
        for pair_data in closest_approach_data:
            sat1 = pair_data['sat1_elements']
            sat2 = pair_data['sat2_elements']
            
            # Calculate features
            altitude_diff = abs(sat1['altitude'] - sat2['altitude'])
            inclination_diff = abs(sat1['inclination'] - sat2['inclination'])
            energy_ratio = sat1['mean_motion'] / sat2['mean_motion']
            
            # Multi-factor collision probability
            collision_prob = self.compute_collision_probability(
                pair_data['closest_approach_km'],
                pair_data['relative_velocity_kms'],
                altitude_diff,
                inclination_diff,
                sat1['eccentricity'],
                sat2['eccentricity']
            )
            
            # Features array (same as before)
            features = np.array([
                collision_prob,
                pair_data['closest_approach_km'],
                pair_data['relative_velocity_kms'],
                altitude_diff,
                inclination_diff,
                energy_ratio,
                sat1['eccentricity'],
                sat2['eccentricity'],
            ])
            
            # Risk classification using same thresholds
            if collision_prob > 1e-4:
                risk_class = 2
            elif collision_prob > 1e-6:
                risk_class = 1
            else:
                risk_class = 0
                
            collision_data.append({
                'sat1_id': pair_data['sat1_id'],
                'sat2_id': pair_data['sat2_id'],
                'features': features,
                'risk_class': risk_class
            })
        
        return collision_data