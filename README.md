# Video Classification in PyTorch 
A PyTorch re-implementation of various deep neural networks on a custom dataset.   
Currently, only 3D Convolutional Neural Networks [[**D. Tran et al., 2015**]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf) is implemented. Various deep neural networks will be implemented in the future.

## Installation
Tested on MacOSX 12.2.1. and Python 3.8.1.
1. Install ffmpeg (https://github.com/FFmpeg/FFmpeg)
    - For linux
        ```shell
        $ apt-get install ffmpeg
        ```
    - For MacOSX
        ```shell
        $ brew install ffmpeg 
        ```
2. Install PyTorch nightly (for installing torchvision from source)
    ```shell
    $ pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html torchaudio
    ```

3. Install torchvision from the source to use VideoReader (it does not work on python3.9.x)
    ```shell
    $ git clone https://github.com/pytorch/vision.git
    ```
    ```shell
    $ python install setup.py 
    ```
    * For MacOSX, 
        - Remove ```'sys.platform != "linux"'``` in line # 349 of setup.py (for detecting ffmpeg)     
        - Then run,
            ```shell 
            $ MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
            ```
4. Install the packages required
    ``` shell
    $ pip install -r requirements.txt
    ```


---
## A brief guide for deploying models with AWS
1. If the size of external libraries is smaller than 250MB,
   - Import libraries in Lambda Function by adding the libraries to a Lambda Layer
     - To do so, compress the libraries into "python.zip" and upload it to the Lambda Layer
2. Otherwise,
   - Use an EC2 or packaging Lambda functions as a container image of up to 10 GB in size
---

## Dataset
A Dataset should follow one of the two structures below. The configuration for a dataset (video path, annotation path, etc.) is defined in config.yaml file.
```
    └── custom_dataset/                        
        ├── video_data/                        
        │   ├── 1.avi                      
        │   └── ...                        
        ├── classInd.txt    -->  consists of 'class_idx' and 'class_name' on one line. class_idx should be 0-based.      
        └── test_train_split/
            ├── trainlist.txt  --> consists of 'class_name/video_data_name.avi' and 'class_idx' on one line.
            └── testlist.txt

               OR

    └── custom_dataset/
            ├── video_data/
            │   ├── class1/
            │   │   ├── class_1_a.avi 
            │   │   └── ...
            │   └── class2/
            │       ├── class_2_a.avi
            │       └── ...
            └── test_train_split/
                ├── trainlist.txt --> consists of 'class_name/video_data_name.avi' and 'class_idx' on one line.
                └── testlist.txt
```

## Run
### Train
```shell
$ python main.py --model_name "C3D" --dataset_name {dataset_name_for_training} --run_mode "train" --resume_epoch {epoch_number} --tb --pretrained_model_path {pretrained_model_path}
```
- dataset_name [required]: the dataset name defined in the config.yaml.
- resume_epoch [optional]: epoch number to resume training.
- tb [optional]: whether to use tensorboard or not.
- pretrained_model_path [optional]: the pre-trained model path when necessary.
- You can change hyperparameters for training in the config.yaml.

### Test
```shell
$ python main.py --model_name "C3D" --dataset_name {dataset_name_for_test} --run_mode "test" --model_path {model_path}
```
- dataset_name [required]: the dataset name defined in the config.yaml.
- model_path [required]: the model path to be used for evaluation.
### Demo
```shell
$ python main.py --model_name "C3D" --video_path {video_path} --run_mode "demo" --model_path {model_path}
```
- video_path [required]: the video file path to be used for a demo.
- model_path [required]: the model path to be used for a demo.

#### Demo examples
ApplyEyeMakeUp             |  BaseballPitch                 |  WalkingWithDog
:-------------------------:|:-------------------------:|:-------------------------:
![output_1](https://user-images.githubusercontent.com/13464650/163706504-51b1372e-9f23-44a9-87ae-13a255934448.gif)         | ![output_2](https://user-images.githubusercontent.com/13464650/163706532-a3e61c64-727d-4293-a20a-61699d1503be.gif)    | ![output_3](https://user-images.githubusercontent.com/13464650/163706537-1e1bf81f-701d-455d-b1e2-9798bf05f311.gif)



### To-Do list
- [ ] Implementation of other Deep Neural Networks
- [ ] Implementation of post-processing