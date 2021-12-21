import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GNNkeras",
    version="2.0",
    author="Niccolò Pancino, Pietro Bongini",
    author_email="niccolo.pancino@unifi.it",
    description="Graph Neural Network and Layered Graph Neural Network Tensorflow 2.x keras-based implementations",
    long_description=long_description,
    url="https://github.com/NickDrake117/GNNKeras/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
