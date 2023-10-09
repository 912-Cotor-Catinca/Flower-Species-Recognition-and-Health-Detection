import glob
import random
import collections
import Augmentor
import numpy as np
import os

root_directory = "C:/Users/Catinca/Documents/TomatoDataset/*"

folders = []
for f in glob.glob(root_directory):
    if os.path.isdir(f):
        folders.append(os.path.abspath(f))

print("Folders (classes) found: %s " % [os.path.split(x)[1] for x in folders])

pipelines = {}
for folder in folders:
    print("Folder %s:" % (folder))
    pipelines[os.path.split(folder)[1]] = (Augmentor.Pipeline(folder))
    print("\n----------------------------\n")

for p in pipelines.values():
    print("Class %s has %s samples." % (p.augmentor_images[0].class_label, len(p.augmentor_images)))

max_nr = 5357

for pipeline in pipelines.values():
    pipeline.rotate(probability=0.75, max_left_rotation=10, max_right_rotation=10)
    pipeline.flip_left_right(probability=0.8)
    pipeline.skew(probability=0.4)
    pipeline.random_distortion(probability=0.5, grid_width=3, grid_height=7,magnitude=2)
    pipeline.crop_centre(probability=0.1,percentage_area=0.8)
    pipeline.sample(max_nr - len(pipeline.augmentor_images))