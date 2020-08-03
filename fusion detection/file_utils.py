import os
def create_folders(folder_name):
    for affix in ["train", "val", "test", "test-challenge"]:
        if not os.path.exists(os.path.join(".", folder_name, affix)):
            os.makedirs(os.path.join(".", folder_name, affix))
            os.makedirs(os.path.join(".", folder_name, affix, "images"))
            os.makedirs(os.path.join(".", folder_name, affix, "annotations"))
