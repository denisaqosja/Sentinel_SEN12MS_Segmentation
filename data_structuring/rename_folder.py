import numpy as np
import os


def rename_folder_names(path):
    season_folders = os.listdir(path)
    for season_folder in season_folders:
        if not any(ext in season_folder for ext in (".tar", ".txt", ".sha", ".csv", ".py")):
            sensor_folder_names = os.listdir(os.path.join(path, season_folder))
            
            for sensor_folder in sensor_folder_names:
                sensor_folder_path = os.path.join(path, season_folder, sensor_folder)
                print(sensor_folder_path)

                sensor, idx = sensor_folder.split("_")
                if len(idx) <3:
                    idx_padded = f"{int(idx):03d}"
                    print(sensor, idx, idx_padded)
        
                    padded_folder_name = f"{sensor}_{idx_padded}"
                    padded_sensor_folder_path = os.path.join(path, season_folder, padded_folder_name)
                    print(padded_sensor_folder_path)

                    # os.rename(sensor_folder_path, padded_sensor_folder_path)
                    
    return 


def rename_file_names(path):
    season_folders = os.listdir(path)
    for season_folder in season_folders:
        if not any(ext in season_folder for ext in (".tar", ".txt", ".sha", ".csv", ".py")):
            sensor_folder_names = os.listdir(os.path.join(path, season_folder))
            
            for sensor_folder in sensor_folder_names:
                sensor_folder_path = os.path.join(path, season_folder, sensor_folder)
                
                for files in os.listdir(sensor_folder_path):
                    root_name, ext = os.path.splitext(files)
                    root_name_parts = root_name.split("_")
                
                    if len(root_name_parts[4]) <4:
                        print(f"{root_name = }, {ext = }")
                        print(f"{root_name_parts = }")
                        
                        s = root_name_parts[4]
                        letter, number = s[0], s[1:]
                        number_padded = f"{int(number):03d}"
                        padded_root_name = f"{root_name_parts[0]}_{root_name_parts[1]}_{root_name_parts[2]}_{root_name_parts[3]}_{letter}{number_padded}{ext}"
                        print(padded_root_name)
                        
                        old_file_path = os.path.join(sensor_folder_path, files)
                        new_file_path = os.path.join(sensor_folder_path, padded_root_name)
                        print(old_file_path, new_file_path)
                        os.rename(old_file_path, new_file_path)
                        
                        
    return 


if __name__ == "__main__":
    path = "C:/Users/denisa.qosja/Documents/Datasets/SEN12/" # "/data/workspace/denqos/Datasets/SEN12MS/"
    
    rename_folder_names(path)
    # rename_file_names(path)
    
    
    
