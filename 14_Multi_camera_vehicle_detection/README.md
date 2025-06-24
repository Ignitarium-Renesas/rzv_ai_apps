# 14_Multi_camera_vehicle_detection

## Application: Overview
This application is used to detect 9 types of vehicles below from camera input.  
Also it can be used for these vehicles at 360 angle with multi cameras.
- Car, policecar, ambulance, bicycle, bus, truck, bike, auto and fire engine

The AI model used for the sample application is [YOLOX-l](https://github.com/Megvii-BaseDetection/YOLOX)

### Targeted product
| Product | Supported AI SDK version |
| ---- | ---- |
| RZ/V2H Evaluation Board Kit (RZ/V2H EVK) | RZ/V2H AI SDK **v5.20** |
| RZ/V2N Evaluation Board Kit (RZ/V2N EVK) | RZ/V2N AI SDK **v6.00** | 

 ### Sample Video for RZ/V2H on Youtube
 <a href="https://youtu.be/gsh461URVpY" target="_blank\">
  <img src="./img/thumbnail.png" alt="Multi Camera vehicle demo" width="400" />
</a>

## Application: Requirements

#### Hardware Requirements
Prepare the following equipments referring to [Getting Started](https://renesas-rz.github.io/rzv_ai_sdk/getting_started).

| Equipment | Details |
| ---- | ---- |
| RZ/V2H, RZ/V2N EVK | Evaluation Board Kit for RZ/V2H, RZ/V2N |
| USB camera | Used as a camera input source. <br> Up to 3cameras.<br> Recommended model number : Logicool c930e |
| MIPI camera | Used as a camera input source. <br> RZ/V2H: Up to 4 cameras.<br> RZ/V2N: Up to 2 cameras.<br> To use MIPI camera, please refer to e-CAM22_CURZH provided by [e-con Systems](https://www.e-consystems.com/renesas/sony-starvis-imx462-ultra-low-light-camera-for-renesas-rz-v2h.asp).|
| HDMI monitor | Used to display the graphics of the board. |
| USB Cable Type-C | Connect AC adapter and the board. |
| HDMI cable | Connect HDMI monitor and RZ/V2H, RZ/V2N Board. |
| AC Adapter | USB Power Delivery adapter for the board power supply.<br>100W is required. |
| microSD Card | Must have over 16GB capacity of blank space.<br>Operating Environment: Transcend UHS-I microSD 300S 16GB |
| Linux PC | Used to build application and setup microSD card.<br>Operating Environment: Ubuntu 20.04 |
| SD card reader | Used for setting up microSD card. |
| USB Hub | Used for connecting USB Mouse and USB Keyboard to the board. <br> Recommended model number : SANWA USB-3H703BK|
| USB Mouse | Used for HDMI screen control. |
| USB Keyboard | Used for terminal input. |
>**Note:**
All external devices will be attached to the board and does not require any driver installation (Plug n Play Type).

Connect the hardware as shown below.  
Regarding MIPI camera, please refer to the user manual of [e-con Systems](https://www.e-consystems.com/renesas/sony-starvis-imx462-ultra-low-light-camera-for-renesas-rz-v2h.asp).

- For using MIPI camera

|RZ/V2H EVK | RZ/V2N EVK |
|:---|:---|
|<img src=./img/hw_conf_mipi_v2h.png width=600>|<img src=./img/hw_conf_mipi_v2n.png width=600> |

- For using USB camera  

|RZ/V2H EVK | RZ/V2N EVK |
|:---|:---|
|<img src=./img/hw_conf_usb_v2h.png width=600>|<img src=./img/hw_conf_usb_v2n.png width=600> |

When using the keyboard connected to RZ/V Evaluation Board, the keyboard layout and language are fixed to English.

## Application: Build Stage

>**Note:** User can skip to the next stage (deploy) if they don't want to build the application. All pre-built binaries are provided.

This project expects the user to have completed [Getting Started](https://renesas-rz.github.io/rzv_ai_sdk/getting_started) provided by Renesas. 

After completion of Getting Started, the user is expected of following conditions.
- The board setup is done.
- SD card is prepared.
- Following docker container is running on the host machine.

   | Board| Docker container |
   | ---- | ---- |
   | RZ/V2H EVK | rzv2h_ai_sdk_container |
   | RZ/V2N EVK | rzv2n_ai_sdk_container |

    >**Note 1:** Docker environment is required for building the application.  
<!--    >**Note 2:** Since RZ/V2N is a brother chip of RZ/V2H, the same environment can be used.  -->

#### Application File Generation
1. On your host machine, download the repository from the GitHub to the desired location. 
    1. It is recommended to download/clone the repository on the `data` folder which is mounted on the docker container as shown below. 
    ```sh
    cd <path_to_data_folder_on_host>/data
    git clone https://github.com/Ignitarium-Renesas/rzv_ai_apps.git
    ```
    > Note 1: Please verify the git repository url if error occurs.

    > Note 2: This command will download whole repository, which include all other applications.<br>
     If you have already downloaded the repository of the same version, you may not need to run this command.
    
2. Run (or start) the docker container and open the bash terminal on the container.  
E.g., for RZ/V2H, use the `rzv2h_ai_sdk_container` as the name of container, created from  `rzv2h_ai_sdk_image` docker image.  
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
    cmake -DCMAKE_TOOLCHAIN_FILE=./toolchain/runtime.cmake ..
    make -j$(nproc)
    ```
6. The following application file would be genarated in the `${PROJECT_PATH}/14_Multi_camera_vehicle_detection/src/build` directory
   - multi_camera_vehicle_detection_app
<!--    >**Note:** Since RZ/V2N is a brother chip of RZ/V2H,  the same source code can be used.  -->


## Application: Deploy Stage
For the ease of deployment all the deployables file and folders are provided in following folder.
|Board | `EXE_DIR` |
|:---|:---|
|RZ/V2H EVK|[exe_v2h](./exe_v2h)  |
|RZ/V2N EVK|[exe_v2n](./exe_v2n)  |
<!-- >**Note:** Since RZ/V2N is a brother chip of RZ/V2H,  the same execution environment can be used.  -->

Each folder contains following items.
|File | Details |
|:---|:---|
|Multi_camera_vehicle_detection_yoloxl | Model object files for deployment.|
|multi_camera_vehicle_detection_app | application file. |

1. Follow the steps below to deploy the project on the board. 

    1. Run the commands below to download the necessary file.
    ```
    cd ${PROJECT_PATH}/14_Multi_camera_vehicle_detection/<EXE_DIR>/Multi_camera_vehicle_detection_yoloxl
    wget <URL>/<SO_FILE>
    ```
    |Board | `EXE_DIR` |`URL` |`SO_FILE` |File Location |
    |:---|:---|:---|:---|:---|
    |RZ/V2H EVK|[exe_v2h](./exe_v2h)  |<span style="font-size: small">`https://github.com/Ignitarium-Renesas/rzv_ai_apps/releases/tag/v5.10`</span>  |<span style="font-size: small">`14_Multi_camera_vehicle_detection_deploy_tvm-v230.so`</span> |[Release v5.10](https://github.com/Ignitarium-Renesas/rzv_ai_apps/releases/tag/v5.10)  |
    |RZ/V2N EVK|[exe_v2n](./exe_v2n)  |<span style="font-size: small">`https://github.com/Ignitarium-Renesas/rzv_ai_apps/releases/tag/v6.00`</span>  |<span style="font-size: small">`14_Multi_camera_vehicle_detection_deploy_tvm_v2n-v251.so`</span> |[Release v6.00](https://github.com/Ignitarium-Renesas/rzv_ai_apps/releases/tag/v6.00)  |

    2. Rename the `14_Multi_camera_vehicle_detection_deploy_tvm*.so` to `deploy.so`.
    ```
    mv <SO_FILE> deploy.so
    ```
    3. Copy the following files to the `/home/root/tvm` directory of the rootfs (SD Card) for the board.
        -  All files in <EXE_DIR> directory. (Including `deploy.so` file.)
        -  `14_Multi_camera_vehicle_detection` application file if you generated the file according to [Application File Generation](#application-file-generation)

2. Folder structure in the rootfs (SD Card) is shown below.<br>
   Check if `libtvm_runtime.so` exists in the rootfs directory (SD card) on the board.
- For RZ/V2H
```sh
├── usr/
│   └── lib64/
│       └── libtvm_runtime.so
└── home/
    └── root/
        └── tvm/ 
            ├── Multi_camera_vehicle_detection_yoloxl/
            │   ├── preprocess
            │   ├── deploy.json
            │   ├── deploy.params
            │   └── deploy.so
            └── multi_camera_vehicle_detection_app
```
   - For RZ/V2N
```sh
├── usr/
│   └── lib/
│       └── libtvm_runtime.so
└── home/
　　└── weston/
　　　　└──  tvm/
            ├── Multi_camera_vehicle_detection_yoloxl/
            │   ├── preprocess
            │   ├── deploy.json
            │   ├── deploy.params
            │   └── deploy.so
            └── multi_camera_vehicle_detection_app
```

>**Note:** The directory name could be anything instead of `tvm`. If you copy the whole `exe_v2h` folder on the board. You are not required to rename it `tvm`.

## Application: Run Stage

1. On the board terminal, go to the `tvm` directory of the rootfs.
   - For RZ/V2H
    ```sh
    cd /home/root/tvm
    ```
   - For RZ/V2N
    ```sh
    cd /home/weston/tvm
    ```

2. Run the application. The 2nd argument (2) means the number of cameras.
   - For RZ/V2H

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
    - For RZ/V2N
    - Application with USB camera input
    ```sh
    su
    ./multi_camera_vehicle_detection_app USB 2
    exit    # After pressing ENTER key to terminate the application.
    ```
    - Application with USB camera input with flip mode
    ```sh
    su
    ./multi_camera_vehicle_detection_app USB 2 FLIP
    exit    # After pressing ENTER key to terminate the application.
    ```
    - Application with MIPI camera input 
    ```sh
    su
    ./multi_camera_vehicle_detection_app MIPI 2 
    exit    # After pressing ENTER key to terminate the application.
    ```
    - Application with MIPI camera input with flip mode
    ```sh
    su
    ./multi_camera_vehicle_detection_app MIPI 2 FLIP
    exit    # After pressing ENTER key to terminate the application.
    ```  
>**Note:** For RZ/V2N AI SDK v6.00 and later, you need to switch to the root user with the 'su' command when running an application.<br>
This is because when you run an application from a weston-terminal, you are switched to the "weston" user, which does not have permission to run the /dev/xxx device used in the application.<br>

3. Following window shows up on HDMI screen*.  
<img src="./img/app_run.png" alt="Sample application output"
     margin-right=10px; 
     width=600px;
     height=334px />  
    - In the case of using 4 MIPI cameras  
<img src="./img/app1_run.png" alt="Sample application output"
     margin-right=10px; 
     width=600px;
    height=334px />

*Performance in the screenshot is for RZ/V2H EVK.

4. To terminate the application, switch the application window to the terminal by using Super(windows key)+ Tab and press ENTER key on the terminal of the board.

## Application: Configuration 
### AI Model
- YOLOX-L: [Megvii-BaseDetection
](https://github.com/Megvii-BaseDetection/YOLOX)  
- Datasets: *[Car1](https://universe.roboflow.com/hungdk-t8jb0/nhandienxeoto-udgcp), *[Car2](https://universe.roboflow.com/project-fjp7n/car-detection-vwdhg), *[policecar1](https://universe.roboflow.com/fyp-tc-idn2o/police-cars-sumfm), *[policecar2](https://universe.roboflow.com/maryam-mahmood-6hoeq/pol-tslhg), *[ambulance1](https://universe.roboflow.com/ambulance-k0z3x/ambulance-detection-azspv), *[ambulance2](https://universe.roboflow.com/school-87zwx/emegency-vehicle-detection), *[bicycle1](https://universe.roboflow.com/vtc-ywqwf/tt-aio6y), *[bicycle2](https://universe.roboflow.com/north-south-university-faox7/bicycle-bdti6), *[bicycle3](https://cocodataset.org/#download), *[bus1](https://universe.roboflow.com/titu/bus-jm7t3), *[bus2](https://universe.roboflow.com/final-year-project-shhpl/bus-detection-2wlyo), *[bus3](https://universe.roboflow.com/fyp-object-detection-tc8af/sya-bus), *[truck](https://images.cv/dataset/garbage-truck-image-classification-dataset), *[bike1](https://universe.roboflow.com/subham-bhansali-fedah/bike-detection-tzvlj), *[fireengine1](https://universe.roboflow.com/grad-project-tjt2u/fire-truck-xumw3) , 
*[fireengine2](https://universe.roboflow.com/pouria-maleki/firetruck), *[fireengine3](https://universe.roboflow.com/hugo-drfu2/-500-pumqz-qhctg/dataset/2) , *[auto1](https://universe.roboflow.com/rutviknirma/smart-traffic-management-system), *[auto2](https://universe.roboflow.com/graduation-project-rtgrc/tuk-tuk-labelling)
  
Input size:   1x3x320x320 \
Output1 size: 1x14x40x40 \
Output2 size: 1x14x20x20\
Output3 size: 1x14x10x10
 
### AI inference time
|Board | AI inference time|
|:---|:---|
|RZ/V2H EVK | Approximately 15ms per 1 camera |
|RZ/V2N EVK | Approximately 30ms per 1 camera |
 
### Processing
 
|Processing | Details |
|:---|:---|
|Pre-processing | Processed by DRP-AI. <br> |
|Inference | Processed by DRP-AI and CPU. |
|Post-processing | Processed by CPU. |

## Reference
- For RZ/V2H, RZ/V2N EVK, this application supports USB camera only with 640x480 resolution.  
FHD resolution is supported by e-CAM22_CURZH camera (MIPI).  
Please refer to following URL for how to change camera input to MIPI camera.  
[https://renesas-rz.github.io/rzv_ai_sdk/latest/about-applications](https://renesas-rz.github.io/rzv_ai_sdk/latest/about-applications#mipi). 
