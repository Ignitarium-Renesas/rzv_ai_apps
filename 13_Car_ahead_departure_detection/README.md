# 13_Car_ahead_departure_detection

## Application: Overview
Car ahead departure detection is a sample application that detects the movement of a car in the front away from a reference point.
Application uses a deep learning based object detector tinyYoloV3 to detect the vehicles and a SORT based tracker to track the objects. 

The AI model used for the sample application is [TINYYOLOV3](https://arxiv.org/pdf/1910.01271.pdf).

#### <ins>Working of SORT tracker</ins>
**<ins>SORT Tracker</ins>** : SORT Tracker is a visual multiple object tracking framework based on rudimentary data association and state estimation techniques. It is designed for online tracking applications where only past and current frames are available and the method produces object identities on the fly.
- Tracker:[SORT](https://github.com/yasenh/sort-cpp)

### Targeted product

 - RZ/V2H Evaluation Board Kit (RZ/V2H EVK)

## Application: Requirements

#### Hardware Requirements
Prepare the following equipments referring to [Getting Started](https://renesas-rz.github.io/rzv_ai_sdk/getting_started).
| Equipment | Details |
| ---- | ---- |
| RZ/V2H EVK | Evaluation Board Kit for RZ/V2H |
| USB camera | - |
| HDMI monitor | Display the application. |
| HDMI cable | Connect HDMI monitor and RZ/V2H Board. |
| microSD Card | Used as filesystem. |
| USB Hub | Used for connecting USB Mouse and USB Keyboard to the board. |
| USB Mouse | Used for HDMI screen control. |
| USB Keyboard | Used for terminal input. |
>**Note:**
All external devices will be attached to the board and does not require any driver installation (Plug n Play Type).

Connect the hardware as shown below.  

<img src="./img/hw_conf_v2h.png" alt="Connected Hardware"
     margin-right=10px; 
     width=600px;
     height=334px />


When using the keyboard connected to RZ/V2H Evaluation Board, the keyboard layout and language are fixed to English.

## Application: Build Stage

>**Note:** User can skip to the next stage (deploy) if they don't want to build the application. All pre-built binaries are provided.

This project expects the user to have completed [Getting Started](https://renesas-rz.github.io/rzv_ai_sdk/getting_started) provided by Renesas. 

After completion of Getting Started, the user is expected of following conditions.
- The board setup is done.
- SD card is prepared.
- The docker container of `rzv2h_ai_sdk_image` is running on the host machine.

>**Note:** Docker environment is required for building the application. 


#### Application File Generation
1. On your host machine, download the repository from the GitHub to the desired location. 
    1. It is recommended to download/clone the repository on the `data` folder which is mounted on the `rzv2h_ai_sdk_container` docker container as shown below. 
    ```sh
    cd <path_to_data_folder_on_host>/data
    git clone https://github.com/Ignitarium-Renesas/rzv_ai_apps.git
    ```
    > Note 1: Please verify the git repository url if error occurs.

    > Note 2: This command will download whole repository, which include all other applications.<br>
     If you have already downloaded the repository of the same version, you may not need to run this command.
    
2. Run (or start) the docker container and open the bash terminal on the container.  
Here, we use the `rzv2h_ai_sdk_container` as the name of container, created from  `rzv2h_ai_sdk_image` docker image.  
    > Note that all the build steps/commands listed below are executed on the docker container bash terminal.  

3. Set your clone directory to the environment variable.  
    ```sh
    export PROJECT_PATH=/drp-ai_tvm/data/rzv_ai_apps
    ```
4. Go to the application source code directory.  
    ```sh
    cd ${PROJECT_PATH}/13_Car_ahead_departure_detection/src
    ```
5. Build the application by following the commands below.  
    ```sh
    mkdir -p build && cd build
    cmake -DCMAKE_TOOLCHAIN_FILE=./toolchain/runtime.cmake -DV2H=ON ..
    make -j$(nproc)
    ```
6. The following application file would be genarated in the `${PROJECT_PATH}/13_Car_ahead_departure_detection/src/build` directory
- car_departure_app


## Application: Deploy Stage
For the ease of deployment all the deployables file and folders are provided on the [exe_v2h](./exe_v2h) folder.

|File | Details |
|:---|:---|
|tinyyolov3_car_ahead | Model object files for deployment.|
|car_departure_app | application file. |

1. Follow the steps below to deploy the project on the board. 
    1. Verify the presence of `deploy.so` file in `${PROJECT_PATH}/13_Car_ahead_departure_detection/exe_v2h/tinyyolov3_car_ahead`
    2. Copy the following files to the `/home/root/tvm` directory of the rootfs (SD Card) for the board.
        -  All files in [exe_v2h](./exe_v2h) directory. (Including `deploy.so` file.)
        -  `13_Car_ahead_departure_detection` application file if you generated the file according to [Application File Generation](#application-file-generation)
    3. Check if `libtvm_runtime.so` is there on `/usr/lib64` directory of the rootfs (SD card) on the board.

2. Folder structure in the rootfs (SD Card) would look like:
```sh
├── usr/
│   └── lib64/
│       └── libtvm_runtime.so
└── home/
    └── root/
        └── tvm/ 
            ├── tinyyolov3_car_ahead/
            │   ├── deploy.json
            │   ├── deploy.params
            │   └── deploy.so
            ├── labels.txt
            └── car_departure_app
```
>**Note:** The directory name could be anything instead of `tvm`. If you copy the whole `exe_v2h` folder on the board. You are not required to rename it `tvm`.

## Application: Run Stage

1. On the board terminal, go to the `tvm` directory of the rootfs.
```sh
cd /home/root/tvm
```
2. Run the application.

   - Application with USB camera input
    ```sh
    ./car_departure_app USB
    ```
3. Following window shows up on HDMI screen.  
<img src="./img/app_run.png" alt="Sample application output"
     margin-right=10px; 
     width=600px;
     height=334px />
        
4. To terminate the application, switch the application window to the terminal by using Super(windows key)+ Tab and press ENTER key on the terminal of the board.

## Application: Configuration 
### AI Model
- TinyYoloV3: [ Official Yolo website](https://pjreddie.com/darknet/yolo/)  
- Dataset: *[cocodataset](https://cocodataset.org/#download)

Input size: 1x3x416x416  
Output1 size: 1x255x13x13 <br>
Output2 size: 1x255x26x26

 
### AI inference time
|Board | AI inference time|
|:---|:---|
|RZ/V2H EVK | Approximately 6ms  |
 
### Processing
 
|Processing | Details |
|:---|:---|
|Pre-processing | Processed by CPU. <br> |
|Inference | Processed by DRP-AI and CPU. |
|Post-processing | Processed by CPU. |

 ## Limitations
- This is a simple sample tutorial application. It is provided for an user to experiment with an object detection model with a very basic tracker algorithm. 
- The tracking only happens within a central virtual space.
- Since we are using pretrained TinyYolov3 the experiment shows lesser accuracy at poor light condition.


**TinyYolov3** :
- Light-weight model : Total number of learnable parameters are less as compared to other yolo models.
- Comparatively lower accuracy performance: Some detections are missed in the challenging environment like fast moving objects, noisy background etc.

**SORT Tracker** :
- Performance is strictly average in case of occlusions.
- Missed detection : In case of missed detection, tracker may not be able to predict the precise location of bounding boxes.

## Reference
- For RZ/V2H EVK, this application supports USB camera only with 640x480 resolution.  
FHD resolution is supported by e-CAM22_CURZH camera (MIPI).  
Please refer to following URL for how to change camera input to MIPI camera.  

[https://renesas-rz.github.io/rzv_ai_sdk/latest/about-applications](https://renesas-rz.github.io/rzv_ai_sdk/latest/about-applications#mipi). 
