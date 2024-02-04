import os
import skimage.io as skio
from pathlib import Path
import shutil


def is_valid_path(path):
    """
    Check if the given path exists.

    Args:
        path (str): Path to be checked.

    Returns:
        bool: True if the path exists, False otherwise.
    """
    return os.path.exists(path)


def path_dataset(main_path, path_train="train", path_test="test"):
    """
    Generate and return paths for the training and testing datasets.

    Args:
        main_path (str): Main directory path.
        path_train (str, optional): Subdirectory for the training dataset. Default is "train".
        path_test (str, optional): Subdirectory for the testing dataset. Default is "test".

    Returns:
        Tuple[str, str]: Tuple containing paths for the training and testing datasets.

    Prints:
        str: Status message indicating whether the main path is valid or not.
    """
    if is_valid_path(main_path):
        print(f"The path {main_path} is valid.")
    else:
        print(f"The path {main_path} is not valid. Please check the specified path.")

    train_data_dir = os.path.join( main_path, path_train )
    test_data_dir = os.path.join( main_path, path_test )

    return train_data_dir, test_data_dir


def load_chest_xray_data( data_directory ):
    """
    Load chest X-ray images and corresponding labels from the specified directory.

    The function reads images with the '.jpeg' extension from each subdirectory in the given data directory.
    It returns a tuple containing a list of images and their corresponding labels.

    Args:
        data_directory (str): Directory containing subdirectories for different classes.

    Returns:
        Tuple[List[np.ndarray], List[str]]: Tuple containing a list of images (as NumPy arrays) and their corresponding labels.

    
    Example:
    - images, labels = load_chest_xray_data('/path/to/chest_xray_data')
    """
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


def reorganize_pneumonia_data(main_dir):
    """
    Reorganize pneumonia data by creating separate folders for virus and bacteria images.

    Args:
        main_dir (str): Main directory containing 'train' and 'test' subdirectories.

    The function reorganizes the pneumonia data within the 'train' and 'test' subsets.
    It creates separate folders for 'PNEUMONIA_VIRUS' and 'PNEUMONIA_BACTERIA', moves virus images
    to the new folder, and renames the 'PNEUMONIA' folder to 'PNEUMONIA_BACTERIA'.
    """
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
