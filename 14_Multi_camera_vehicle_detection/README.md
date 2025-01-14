# 14_Multi_camera_vehicle_detection

## Application: Overview
This application is used to detect 10 types of vehicles below from camera input.  
Also it can be used for these vehicles at 360 angle with multi cameras.
- Car, policecar, ambulance, bicycle, bus, truck, bike, tractor , auto and fire engine

The AI model used for the sample application is [TINYYOLOV3](https://pjreddie.com/darknet/yolo/).

### Targeted product

 - RZ/V2H Evaluation Board Kit (RZ/V2H EVK)

 ### Sample Video
 <a href="https://youtu.be/Ft-BGWEu5bY" target="_blank\">
  <img src="./img/thumbnail.png" alt="Multi Camera vehicle demo" width="400" />
</a>

## Application: Requirements

#### Hardware Requirements
Prepare the following equipments referring to [Getting Started](https://renesas-rz.github.io/rzv_ai_sdk/getting_started).

| Equipment | Details |
| ---- | ---- |
| RZ/V2H EVK | Evaluation Board Kit for RZ/V2H |
| USB camera | Up to 3cameras.<br> Recommended model number : Logicool c930e |
| MIPI camera | Up to 4 cameras.<br> To use MIPI camera, please refer to e-CAM22_CURZH provided by [e-con Systems](https://www.e-consystems.com/renesas/sony-starvis-imx462-ultra-low-light-camera-for-renesas-rz-v2h.asp).|
| HDMI monitor | Display the application. |
| HDMI cable | Connect HDMI monitor and RZ/V2H Board. |
| microSD Card | Used as filesystem. |
| USB Hub | Used for connecting USB Mouse and USB Keyboard to the board. |
| USB Mouse | Used for HDMI screen control. |
| USB Keyboard | Used for terminal input. |
>**Note:**
All external devices will be attached to the board and does not require any driver installation (Plug n Play Type).

Connect the hardware as shown below.  
Regarding MIPI camera, please refer to the user manual of [e-con Systems](https://www.e-consystems.com/renesas/sony-starvis-imx462-ultra-low-light-camera-for-renesas-rz-v2h.asp).

- For using MIPI camera  
<img src="./img/hw_conf_mipi_v2h.png" alt="Connected Hardware"
     margin-right=10px; 
     width=600px;
     height=334px />
- For using USB camera  
<img src="./img/hw_conf_usb_v2h.png" alt="Connected Hardware"
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
    cd ${PROJECT_PATH}/14_Multi_camera_vehicle_detection/src
    ```
5. Build the application by following the commands below.  
    ```sh
    mkdir -p build && cd build
    cmake -DCMAKE_TOOLCHAIN_FILE=./toolchain/runtime.cmake -DV2H=ON ..
    make -j$(nproc)
    ```
6. The following application file would be genarated in the `${PROJECT_PATH}/14_Multi_camera_vehicle_detection/src/build` directory
- multi_camera_vehicle_detection_app


## Application: Deploy Stage
For the ease of deployment all the deployables file and folders are provided on the [exe_v2h](./exe_v2h) folder.

|File | Details |
|:---|:---|
|Multi_camera_vehicle_detection_tinyyolov3 | Model object files for deployment.|
|multi_camera_vehicle_detection_app | application file. |

1. Follow the steps below to deploy the project on the board. 
    1. Verify the presence of `deploy.so` file in `${PROJECT_PATH}/14_Multi_camera_vehicle_detection/exe_v2h/Multi_camera_vehicle_detection_tinyyolov3`
    
    2. Copy the following files to the `/home/root/tvm` directory of the rootfs (SD Card) for the board.
        -  All files in [exe_v2h](./exe_v2h) directory. (Including `deploy.so` file.)
        -  `14_Multi_camera_vehicle_detection` application file if you generated the file according to [Application File Generation](#application-file-generation)
    3. Check if `libtvm_runtime.so` is there on `/usr/lib64` directory of the rootfs (SD card) on the board.

2. Folder structure in the rootfs (SD Card) would look like:
```sh
├── usr/
│   └── lib64/
│       └── libtvm_runtime.so
└── home/
    └── root/
        └── tvm/ 
            ├── Multi_camera_vehicle_detection_tinyyolov3/
            │   ├── preprocess
            │   ├── deploy.json
            │   ├── deploy.params
            │   └── deploy.so
            └── multi_camera_vehicle_detection_app
```
>**Note:** The directory name could be anything instead of `tvm`. If you copy the whole `exe_v2h` folder on the board. You are not required to rename it `tvm`.

## Application: Run Stage

1. On the board terminal, go to the `tvm` directory of the rootfs.
```sh
cd /home/root/tvm
```
2. Run the application. The 2nd argument (2) means the number of cameras.

   - Application with USB camera input
    ```sh
    ./multi_camera_vehicle_detection_app USB 2
    ```
    - Application with USB camera input with flip mode
    ```sh
    ./multi_camera_vehicle_detection_app USB 2 FLIP
    ```
    - Application with MIPI camera input 
    ```sh
    ./multi_camera_vehicle_detection_app MIPI 2 
    ```
    - Application with MIPI camera input with flip mode
    ```sh
    ./multi_camera_vehicle_detection_app MIPI 2 FLIP
    ```
   
3. Following window shows up on HDMI screen.  
<img src="./img/app_run.png" alt="Sample application output"
     margin-right=10px; 
     width=600px;
     height=334px />  
    - In the case of using 4 MIPI cameras  
<img src="./img/app1_run.png" alt="Sample application output"
     margin-right=10px; 
     width=600px;
    height=334px /> 
        
4. To terminate the application, switch the application window to the terminal by using Super(windows key)+ Tab and press ENTER key on the terminal of the board.

## Application: Configuration 
### AI Model
- TINYYOLOv3: [Darknet](https://pjreddie.com/darknet/yolo/)  
- Datasets: *[Car1](https://universe.roboflow.com/hungdk-t8jb0/nhandienxeoto-udgcp), *[Car2](https://universe.roboflow.com/project-fjp7n/car-detection-vwdhg), *[policecar1](https://universe.roboflow.com/fyp-tc-idn2o/police-cars-sumfm), *[policecar2](https://universe.roboflow.com/maryam-mahmood-6hoeq/pol-tslhg), *[ambulance1](https://universe.roboflow.com/ambulance-k0z3x/ambulance-detection-azspv), *[ambulance2](https://universe.roboflow.com/school-87zwx/emegency-vehicle-detection), *[bicycle1](https://universe.roboflow.com/vtc-ywqwf/tt-aio6y), *[bicycle2](https://universe.roboflow.com/north-south-university-faox7/bicycle-bdti6), *[bicycle3](https://cocodataset.org/#download), *[bus1](https://universe.roboflow.com/titu/bus-jm7t3), *[bus2](https://universe.roboflow.com/final-year-project-shhpl/bus-detection-2wlyo), *[bus3](https://universe.roboflow.com/fyp-object-detection-tc8af/sya-bus), *[truck](https://images.cv/dataset/garbage-truck-image-classification-dataset), *[bike1](https://universe.roboflow.com/subham-bhansali-fedah/bike-detection-tzvlj), *[bike2](https://universe.roboflow.com/fyp-object-detection-tc8af/sya-bike)
*[tractor](https://images.cv/dataset/tractor-image-classification-dataset), *[fireengine1](https://universe.roboflow.com/grad-project-tjt2u/fire-truck-xumw3) , 
*[fireengine2](https://universe.roboflow.com/pouria-maleki/firetruck), *[auto1](https://universe.roboflow.com/rutviknirma/smart-traffic-management-system), *[auto2](https://universe.roboflow.com/graduation-project-rtgrc/tuk-tuk-labelling)
  
Input size: 1x3x416x416  
Output1 size: 1x45x13x13  
Output2 size: 1x45x26x26 
 
### AI inference time
|Board | AI inference time|
|:---|:---|
|RZ/V2H EVK | Approximately 5ms per 1 camera |
 
### Processing
 
|Processing | Details |
|:---|:---|
|Pre-processing | Processed by DRP-AI. <br> |
|Inference | Processed by DRP-AI and CPU. |
|Post-processing | Processed by CPU. |

