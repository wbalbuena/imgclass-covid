# COVID-19 Machine Learning Image Classifier
## Description
Simple machine learning project to get more familiar with machine learning.

Utilizing Python and TensorFlow, I refined Inception v3, a pre-trained machine learning model, to identify x-rays of lungs with COVID-19.  

I trained the model with a dataset of xrays of patients with COVID, Viral Pneumonia, and healthy lungs.  I also wrote detailed comments to later reference on future machine learning-related projects.

The model can be further refined by adding more images on the training dataset, adding images for validation testing, as well as refining the training parameters to get more accurate results.

## Tools Used
* Python
* TensorFlow
* Inception v3
* Git Bash
* Sublime

## How to Install and Run
1. Install [Python 3](https://www.python.org/downloads/)
2. Clone repository, then enter
``` 
unix/win> git clone https://github.com/wbalbuena/imgclass-covid.git
unix/win> cd imgclass-covid
```
3. Install [TensorFlow](https://www.tensorflow.org/install)
4. (Optional) Adjust the settings on the model.  Train by running model_training.py
```
windows> python model_training.py
```
6. (Optional) Change the folder_path variable in the app.py file to which dataset you want to run the test on.  By default it is set to the Covid dataset.
7. Run the application
```
windows> python app.py
```
