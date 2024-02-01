import os
import skimage.io as skio
from pathlib import Path
import shutil

"""
DATASET PATH
"""
def is_valid_path(path):
    return os.path.exists(path)

def path_dataset(main_path, path_train="train", path_test="test"):
    if is_valid_path(main_path):
        print(f"The path {main_path} is valid.")
    else:
        print(f"The path {main_path} is not valid. Please check the specified path.")

    train_data_dir = os.path.join( main_path, path_train )
    test_data_dir = os.path.join( main_path, path_test )

    return train_data_dir, test_data_dir

"""
LOAD DATA
"""
def load_chest_xray_data( data_directory ):
    dirs = [ d for d in os.listdir( data_directory ) 
            if os.path.isdir( os.path.join( data_directory, d ) ) ]

    labels = []
    images = []
    for d in dirs:
        label_dir = os.path.join( data_directory, d )
        file_names = [ os.path.join( label_dir, f ) 
                        for f in os.listdir( label_dir )
                        if f.endswith(".jpeg") ]

        for f in file_names:
            images.append( skio.imread(f) )
            labels.append( str(d) )
    
    return images, labels

"""
REORGANIZE DATA
"""
def reorganize_pneumonia_data(main_dir):
    for subset in ['train', 'test']:
        pneumonia_folder = os.path.join(main_dir, subset, 'PNEUMONIA')
        virus_folder = os.path.join(main_dir, subset, 'PNEUMONIA_VIRUS')
        bacteria_folder = os.path.join(main_dir, subset, 'PNEUMONIA_BACTERIA')

        # Create 'PNEUMONIA_VIRUS' folder if not exists
        if not os.path.exists(virus_folder):
            os.makedirs(virus_folder)

        # Moving files to new folder
        virus_files = list(Path(pneumonia_folder).glob('VIRUS*.jpeg'))
        for sample in virus_files:
            shutil.move(sample, os.path.join(virus_folder, sample.name))

        # Rename 'PNEUMONIA' to 'PNEUMONIA_BACTERIA'
        os.rename(pneumonia_folder, bacteria_folder)
