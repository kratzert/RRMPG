# -*- coding: utf-8 -*-
# This file is part of RRMPG.
#
# RRMPG is free software with the aim to provide a playground for experiments
# with hydrological rainfall-runoff-models while achieving competitive
# performance results.
#
# You should have received a copy of the MIT License along with RRMPG. If not,
# see <https://opensource.org/licenses/MIT>
import os
import pandas as pd
import numpy as np

from pathlib import Path

CAMELS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data',
                          'camels')

class CAMELSLoader(object):
    """Interface for loading basin data from the CAMELS dataset.
    
    This class provides an easy to use interface to load different basins from
    the CAMELS [1] dataset provided within this Python package. CAMELS stands 
    for Catchment Attributes for Large-Sample Studies and is a hydrological 
    dataset provided by NCAR for 671 catchments in the USA. The data entire 
    data can be downloaded for free at [2]. Within this package we provide the 
    data of just a few catchments as toy data for this package.
    
    [1] Addor, N., A.J. Newman, N. Mizukami, and M.P. Clark, 2017: The CAMELS 
    data set: catchment attributes and meteorology for large-sample studies. 
    version 2.0. Boulder, CO: UCAR/NCAR. doi:10.5065/D6G73C3Q
    
    [2] https://ncar.github.io/hydrology/datasets/CAMELS_attributes
    
    """
    def __init__(self):
        pass
    
    def load_basin(self, basin_number):
        """Load basin with all its data into pandas Dataframe."""
        met_file = f"{basin_number}_lump_cida_forcing_leap.txt"
        streamflow_file = f"{basin_number}_streamflow_qc.txt"
        
        # create full path objects to files
        met_file = Path(CAMELS_DIR) / met_file
        streamflow_file = Path(CAMELS_DIR) / streamflow_file
        
        # read metorological input file
        df = pd.read_csv(met_file, sep='\s+', header=3)
        
        # create datetime index
        dates = df.Year.map(str) +'/'+ df.Mnth.map(str) +'/'+ df.Day.map(str)
        df.index = pd.to_datetime(dates, format="%Y/%m/%d")
        
        # read area from thrid line
        area = pd.read_csv(met_file, header=None, skiprows=2, 
                           nrows=1)[0].tolist()[0]
        
        # load streamflow data and copy qobs in the dataframe from above
        col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
        df2 = pd.read_csv(streamflow_file, sep='\s+', names=col_names)
        dates = df2.Year.map(str) +'/'+ df2.Mnth.map(str) +'/'+ df2.Day.map(str)
        df2.index = pd.to_datetime(dates, format="%Y/%m/%d")
        
        # convert from cubic feet per second to mm/d
        df['QObs(mm/d)'] = df2['QObs']*(0.3048**3)*1000*86400/area
                
        # only return values of complete hydrological years
        start_date = pd.to_datetime(f"{df.Year[0]}/09/30", format="%Y/%m/%d")
        end_date = pd.to_datetime(f"{df.Year[-1]}/09/30", format="%Y/%m/%d")
            
        return df[start_date:end_date]
        
    def get_basin_numbers(self):
        """Return a list of all available basin numbers."""
        files = [f[:8] for f in os.listdir(CAMELS_DIR)]
        return list(set(files))
    
    def get_random_basin(self):
        """Load a basin of the provided data randomly."""
        basin_numbers = self.get_basin_numbers()
        num = np.random.randint(0, len(basin_numbers))
        print(f"Loading data of basin {num}.")
        return self.load_basin(num)