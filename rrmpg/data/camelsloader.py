# -*- coding: utf-8 -*-
# This file is part of RRMPG.
#
# RRMPG is free software with the aim to provide a playground for experiments
# with hydrological rainfall-runoff-models while achieving competitive
# performance results.
#
# You should have received a copy of the MIT License along with RRMPG. If not,
# see <https://opensource.org/licenses/MIT>
import pandas as pd

from pathlib import Path

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
    VALID_BASINS = ['01031500']
    
    def __init__(self):
        pass
    
    def load_basin(self, basin_number):
        """Load basin data pandas Dataframe.
        
        Load the meteorological data, as well as observed discharge and modeled
        potential evapotranspiration of the specified basin from the CAMELS
        data set.
        
        Args:
            basin_number: String of the basin number that shall be loaded.
        
        Returns:
            A pandas DataFrame with the data of the basin.
            
        Raises:
            ValueError: If the basin number is an invalid number. Check the
                .get_basin_numbers() function for a list of all available 
                basins.
        """
        if basin_number not in self.VALID_BASINS:
            msg = [f"Invalid basin number {basin_number}. Must be one of ",
                   f"{self.VALID_BASINS}."]
            raise ValueError("".join(msg))
        
        # Path object to data folder
        data_dir = Path(__file__).parent / 'data' / 'camels'
        
        # Path object to the two needed text files
        met_file = data_dir / f"{basin_number}_lump_cida_forcing_leap.txt"
        streamflow_file = data_dir / f"{basin_number}_05_model_output.txt"
        
        # read metorological input file
        df = pd.read_csv(met_file, sep='\s+', header=3)
        
        # create datetime index
        dates = df.Year.map(str) +'/'+ df.Mnth.map(str) +'/'+ df.Day.map(str)
        df.index = pd.to_datetime(dates, format="%Y/%m/%d")
        
        
        # load model output data, which contains normalized qobs
        df2 = pd.read_csv(streamflow_file, sep='\s+', header=0)
        dates = df2.YR.map(str) +'/'+ df2.MNTH.map(str) +'/'+ df2.DY.map(str)
        df2.index = pd.to_datetime(dates, format="%Y/%m/%d")
        
        # copy qobs and pet
        df['PET'] = df2['PET']
        df['QObs(mm/d)'] = df2['OBS_RUN']
 
        # drop unnecessary columns
        df = df.drop(['Year', 'Mnth', 'Day', 'Hr'], axis=1)
                
        # only return values of complete hydrological years
        start_date = pd.to_datetime(f"{df.index[0].year}/10/01", 
                                    format="%Y/%m/%d")
        end_date = pd.to_datetime(f"{df.index[-1].year}/09/30", 
                                  format="%Y/%m/%d")
            
        return df[start_date:end_date]
        
    def get_basin_numbers(self):
        """Return a list of all available basin numbers."""            
        return self.VALID_BASINS