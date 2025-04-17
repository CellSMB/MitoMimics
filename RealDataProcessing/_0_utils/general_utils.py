import os

def create_folder(folder_path):
    """
    Create a folder at the specified path.
    
    Parameters:
    folder_path (str): The path to the folder to create.
    
    Returns:
    None: If the folder is created successfully or already exists and is empty.
    
    Raises:
    Exception: If the folder already exists and is not empty.
    """
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created: {folder_path}")
        else:
            if not os.listdir(folder_path):
                print(f"Folder already exists and is empty: {folder_path}")
            else:
                raise Exception(f"Folder already exists and is not empty: {folder_path}")
    except Exception as e:
        print(e)

