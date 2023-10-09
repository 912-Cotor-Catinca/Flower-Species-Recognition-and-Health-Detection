import os
import glob

root_directory = "C:/Users/Catinca/Documents/DataSet/*"

folders = []
for f in glob.glob(root_directory):
    if os.path.isdir(f):
        folders.append(os.path.abspath(f))

for folder in folders:
    print(folder)
    output_folders = folder + "\output"
    print(output_folders)
    for file in os.listdir(output_folders):
        print(file)
        os.remove(output_folders + "\\" +file)

for folder in folders:
    print(folder)
    output_folders = folder + "\output"
    os.rmdir(output_folders)

# src_path = "C:/Users/Catinca/Documents/DataSet/"
# sclasses = ["leaf1", 'leaf2', 'leaf3', 'leaf4', 'leaf5', 'leaf6', 'leaf7', 'leaf8', 'leaf9', 'leaf10', 'leaf11',
#             'leaf12', 'leaf13', 'leaf14', 'leaf15']
#
# import re
#
#
# def sorted_alphanumeric(data):
#     convert = lambda text: int(text) if text.isdigit() else text.lower()
#     alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
#     return sorted(data, key=alphanum_key)
#
# dirlist = sorted_alphanumeric(os.listdir(src_path))
# print(dirlist)
#
# for sclass in dirlist:
#     print(sclass)
#     path = os.path.join(src_path, sclass)
#     class_num = sclasses.index(sclass)
#     print(class_num)
