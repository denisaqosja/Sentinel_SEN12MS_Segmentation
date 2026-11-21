import os
import torch 
import argparse
import numpy as np
import tifffile as tiff
from PIL import Image


DATASET = "SEN12MS" # "DFC2020"  # SEN12MS


SEASONS_SEN12MS = {"spring": "ROIs1158_spring", 
          "summer": "ROIs1868_summer", 
          "fall":   "ROIs1970_fall", 
          "winter": "ROIs2017_winter"
         } 

SEASONS_DFC = {"spring": "ROIs0000_spring", 
            "summer": "ROIs0000_summer", 
            "fall":   "ROIs0000_autumn", 
            "winter": "ROIs0000_winter"
            } 

SEASONS = SEASONS_SEN12MS if DATASET == "SEN12MS" else SEASONS_DFC
SENSORS = ["s1", "s2", "lc"] if DATASET == "SEN12MS" else ["s1", "s2", "lc", "dfc"]


def get_arguments():
    """
    Obtain the arguments given from terminal; 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", type=str, help="The path where the data is located")
    args = parser.parse_args()
    
    return args


class Dataset:
    def __init__(self, datapath):
        super().__init__()
        self.datapath = datapath
        
        print("Loading the data")
        self.s1_paths, self.s2_paths, self.lc_paths, self.dfc_paths = self.get_data_directories(dir_only_based_on_sensor=True)  
        return 
        
    
    def read_tif(self, filepath):
        tif_data = tiff.imread(filepath)
        return tif_data
    
    
    def get_tif_dir(self, path, season, sensor, sensor_file, data_dict):
        sensor_file_path = os.path.join(path, sensor_file) # e.g. sensor_file = s1_100
        for tif_file in sorted(os.listdir(sensor_file_path)): 
            data_dict[sensor][season].append(os.path.join(path, sensor_file, tif_file))
        return 
    
    
    def get_data_directories(self, dir_only_based_on_sensor=True):
        """
        To load the data in this function, a dictionary that separates the directories based on the sensor (s1, s2, lc) but also season. 
        If dir_only_based_on_sensor is set to True, the dictionary that separates the paths based on seasons is converted to a simple list without season distinguishment.
        
        The sensor_dir is a dictionary of shape: 
        dict = {"s1": 
                    {
                        "spring": ["dir1", "dir2", ....], 
                        ...
                        ...
                        "winter": ["dir1", "dir2", ....] 
                    },
                "s2": 
                    {
                        "spring": ["dir1", "dir2", ....], 
                        ...
                        ...
                        "winter": ["dir1", "dir2", ....] 
                    },
                "lc": 
                    {
                        "spring": ["dir1", "dir2", ....], 
                        ...
                        ...
                        "winter": ["dir1", "dir2", ....] 
                    }
                }
        
        Args: 
            -  dir_only_based_on_sensor: Bool
        
        Return: 
            - directiory of data files, 
            data_dir_sensor = {"s1": ["spring_dir1", "spring_dir2", ..., "summer_dir1", ......, "winter_dir"],
                    "s2": ["spring_dir1", "spring_dir2", ..., "summer_dir1", ......, "winter_dir"],
                    "lc": ["spring_dir1", "spring_dir2", ..., "summer_dir1", ......, "winter_dir"]
                    }
        """
       
        # the data is to be saved as dict: 
        sensor_dict = {sensor: {season: [] for season in SEASONS.keys()} for sensor in SENSORS}
        
        for season, season_dir in SEASONS.items():
            path = os.path.join(self.datapath, season_dir)
            sensor_data = sorted(os.listdir(path))
            for sensor_file in sensor_data:
                for sensor in SENSORS:  
                    if sensor in sensor_file:
                        self.get_tif_dir(path, season, sensor, sensor_file, sensor_dict)
        
        # convert the dict of 
        if dir_only_based_on_sensor:
            data_dir_sensor = {
                    key: [v for sublist in subdict.values() for v in sublist]
                    for key, subdict in sensor_dict.items()
                }

            sensor_dict = data_dir_sensor
            
        s1_dict = sensor_dict["s1"]
        s2_dict = sensor_dict["s2"]
        lc_dict = sensor_dict["lc"]
        dfc_dict = sensor_dict["dfc"] if DATASET == "DFC2020" else None
        
        return s1_dict,  s2_dict, lc_dict, dfc_dict


    def write_tiff(self, data, save_dir, f_name):
        """ Save the data as tif -> memory efficient"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        tiff.imwrite(os.path.join(save_dir, f_name), data, compression='zlib', bigtiff=True)
        
        return


    def load_and_save_sensor_data(self, data_paths, save_dir, sensor_type="s1"):
        for i, sensor_file in enumerate(data_paths):
            if "tif.aux.xml" in sensor_file:
                continue
            # obtain the filename from the path
            f_name = os.path.basename(sensor_file)
            print(f"Loading the data ... {f_name}")
            img_tiff = tiff.imread(sensor_file)
            if sensor_type == "s2":
                # RGB = [B04, B03, B02]
                img_tiff = img_tiff[:, :, [3, 2, 1]]   # since the data is indexed from 0, extract the RGB image from channels [3, 2, 1]
            # print(f"Saving the data ... {f_name}")
            self.write_tiff(img_tiff, os.path.join(save_dir, sensor_type), f_name=f_name)

    
    def restructure_data(self):
        """Load the whole data"""
        
        # the SAR images from S1 contain 2 polarizations: VV and VH; therefore their shape is (256, 256, 2)
        self.load_and_save_sensor_data(self.s1_paths, save_dir, sensor_type="s1")
        self.load_and_save_sensor_data(self.s2_paths, save_dir, sensor_type="s2")
        self.load_and_save_sensor_data(self.lc_paths, save_dir, sensor_type="lc")
        self.load_and_save_sensor_data(self.dfc_paths, save_dir, sensor_type="dfc") if DATASET == "DFC2020" else None

        return
    

    
if __name__ == "__main__":
    # datapath = "/data/datasets/SEN12MS/"
    # datapath = "/data/workspace/denqos/Projects/S2_Segmentation/Datasets/DFC_Public_Dataset/"

    args = get_arguments()
    datapath = args.data_directory

    save_dir = os.path.join(datapath, "sensors_v2")

    SentinelDataset = Dataset(datapath)
    SentinelDataset.restructure_data()
    print(f"The whole data is saved!")
    
    