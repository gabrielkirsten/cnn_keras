# __Cnn Keras Image Classifier__ (with NVIDIA cuda)  
__autor__: Gabriel Kirsten Menezes (gabriel.kirsten@hotmail.com)  
__version__: 0.1.1  

A simple image classifier built with __Keras__ using __cuda__ libraries.  

### Requirements:
__You must Install:__  

1. [Python 2.7](https://www.python.org/downloads/);
2. [Nvidia cuda libraries](https://developer.nvidia.com/cuda-downloads);
3. [Nvidia cuDCNN libraries](https://developer.nvidia.com/cudnn);
4. [Tensorflow](https://www.tensorflow.org/install/) or [Theano](http://deeplearning.net/software/theano/install.html)\*;
5. [Keras](https://keras.io/#installation);

**note**:  
\* never tested on Theano.

### How to use:
- Install requirements above;
- Prepare the images for processing (the script allows .png imgages, there are some scripts thats can help with tasks to prepare the image database in folder *utils*, please read the directory structure);
- Run __train.py__ python script;
```
$ sudo python ./train.py
```
- Run __predict.py__ python script;
```
$ sudo python ./predict.py -i "image_to_predict" -cl "class"
```
*(you must specify the class names and image file to be predicted, please read the --help)*

### Directory structure

        .  
        |-- cnn_keras  
        |   |-- data (database images)  
        |   |   |-- train (training images directory)  
        |   |   |   |-- .gitignore (git ignore)
        |   |   |-- validation (validation images directory)  
        |   |   |   |-- .gitignore (git ignore)
        |   |-- src (source files)  
        |   |   |-- classification (classification package)  
        |   |   |   |-- train.py (train source code)  
        |   |   |   |-- predict.py (predict source code  
        |   |-- utils (some scripts)  
        |   |   |-- script_convertall.py (script that converts tiff to png in database image folders)  
        |   |   |-- script_split_data.py (script that splits images between train and validation in database image folders)  
        |-- .gitignore (git ignore)  
        |-- README.md (some infos)  
