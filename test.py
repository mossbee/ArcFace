import os

dataset_folder = '/kaggle/input/ndtwin-train/ndtwin'

folder_num = 0
image_num = 0
for item in os.listdir(dataset_folder):
    if os.path.isdir(os.path.join(dataset_folder, item)):
        folder_num += 1
        for image in os.listdir(os.path.join(dataset_folder, item)):
            if image.endswith('.jpg') or image.endswith('.png') or image.endswith('.jpeg'):
                image_num += 1
            else:
                print(f"Image {image} is not a valid image")
print(f"Folder number: {folder_num}")
print(f"Image number: {image_num}")
