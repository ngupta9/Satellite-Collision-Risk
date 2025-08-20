'''
data_acquisition.py
This module is responsible for acquiring TLE (Two-Line Element) data from the SpaceTrack API.
'''
from spacetrack import SpaceTrackClient

class TLEDataCollector:
    '''
    This class is responsible for collecting TLE (Two-Line Element) data for active satellites
    using the SpaceTrack API.
    '''
    def __init__(self, username, password):
        '''
        Initialize the TLEDataCollector with SpaceTrack API credentials.
        Args:
            username (str): The username for SpaceTrack API.
            password (str): The password for SpaceTrack API.
        Returns:
            None
        '''
        self.st = SpaceTrackClient(identity=username, password=password)
    
    def get_active_satellites(self, limit=1000, days_back=30):
        """
        Get TLE data for recently active satellites
        Args:
            limit (int): The maximum number of satellites to return.
            days_back (int): Number of days back to look for recent TLE data.
        Returns:
            data (str): Raw TLE data for active satellites.
        """
        data = self.st.tle_latest(
            iter_lines=False,
            orderby=['epoch desc'],  # Order by most recent epoch
            limit=limit,
            format='tle',
            epoch=f'>now-{days_back}'  # Only satellites with TLEs from last X days
        )
        return data
    
    def parse_tle_data(self, tle_data):
        """
        Parse raw TLE data into list of (norad_id, line1, line2) tuples
        Args:
            tle_data (str): The raw TLE data as a string.
        Returns:
            tle_records (list): A list of 3-tuples containing (norad_id, line1, line2) 
            for each satellite.
        """
        lines = tle_data.strip().split('\n')
        tle_records = []

        for i in range(0, len(lines), 2):  # Step by 2s: pairs of lines
            if i + 1 < len(lines):
                line1 = lines[i].strip()     # TLE line 1
                line2 = lines[i + 1].strip() # TLE line 2
                
                # Extract NORAD ID from line 1 (positions 3-7)
                norad_id = line1[2:7].strip()
                
                tle_records.append((norad_id, line1, line2))

        return tle_records