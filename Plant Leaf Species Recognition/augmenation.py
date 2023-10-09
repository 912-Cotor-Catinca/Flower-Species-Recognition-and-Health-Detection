import Augmentor
import numpy as np
import os
import glob
import random
import collections

root_directory = "C:/Users/Catinca/Documents/DataSet/*"

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

for pipeline in pipelines.values():
    pipeline.rotate(probability=0.75, max_left_rotation=10, max_right_rotation=10)
    pipeline.flip_left_right(probability=0.8)
    pipeline.skew(probability=0.4)
    pipeline.random_distortion(probability=0.5, grid_width=3, grid_height=7,magnitude=2)
    pipeline.crop_centre(probability=0.1,percentage_area=0.8)
    pipeline.sample(333)

integer_labels = {
    'leaf1': 0,
    'leaf2': 1,
    'leaf3': 2,
    'leaf4': 3,
    'leaf5': 4,
    'leaf6': 5,
    'leaf7': 6,
    'leaf8': 7,
    'leaf9': 8,
    'leaf10': 9,
    'leaf11': 10,
    'leaf12': 11,
    'leaf13': 12,
    'leaf14': 13,
    'leaf15': 14
}

PipelineContainer = collections.namedtuple(
    'PipelineContainer',
    'label label_integer label_categorical pipeline generator'
)

pipeline_containers = []

for label, pipeline in pipelines.items():
    label_categorical = np.zeros(len(pipelines), dtype=int)
    label_categorical[integer_labels[label]] = 1
    pipeline_containers.append(
        PipelineContainer(
            label,
            integer_labels[label],
            label_categorical,
            pipeline,
            pipeline.keras_generator(batch_size=1)
        )
    )


def multi_generator(pipeline_containers, batch_size):
    X = []
    y = []
    for i in range(batch_size):
        pipeline_container = random.choice(pipeline_containers)
        image, _ = next(pipeline_container.generator)
        image = image.reshape((224, 224, 3))  # Or (1, 28, 28) for channels_first, see Keras' docs.
        X.append(image)
        y.append(pipeline_container.label_categorical)  # Or label_integer if required by network
        X = np.asarray(X)
        y = np.asarray(y)
    yield X, y


multi_generator(pipeline_containers, 1)