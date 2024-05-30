/*
 * Original Code (C) Copyright Renesas Electronics Corporation 2023
 *　
 *  *1 DRP-AI TVM is powered by EdgeCortix MERA(TM) Compiler Framework.
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */

/***********************************************************************************************************************
* File Name    : define.h
* Version      : 1.1.0
* Description  : DRP-AI TVM[*1] Application for Human gaze

***********************************************************************************************************************/
#ifndef DEFINE_MACRO_H
#define DEFINE_MACRO_H

/*****************************************
* includes
******************************************/
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>
#include <vector>
#include <map>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <float.h>
#include <atomic>
#include <semaphore.h>
#include <math.h>


/*****************************************
* Static Variables for YOLOv3
* Following variables need to be changed in order to custormize the AI model
******************************************/
/*****************************************
* YOLOv3
******************************************/
/* Model Binary */
const static std::string model_dir = "human_gaze_tinyyolov2";
/* Pre-processing Runtime Object */
const static std::string pre_dir = model_dir + "/preprocess";

/* Label list file */
const static std::string label_list = "labels.txt";

/* Empty List to store label list */
static std::vector<std::string> label_file_map = {};

/* DRP-AI memory offset for model object file*/
#define DRPAI_MEM_OFFSET            (0x2000000)
#define DRPAI_MEM_OFFSET1           (0x0000000)


/*****************************************
 * Macro for YOLOv3
 ******************************************/
/* Number of class to be detected */
#define NUM_CLASS                   (1)
/* Number for [region] layer num parameter */
#define NUM_BB                      (5)
#define NUM_GRID_X                  (13)
#define NUM_GRID_Y                  (13)
#define INF_OUT_SIZE_TINYYOLOV2     (84500)

const static double anchors[] =
{
    1.08,   1.19,
    3.42,   4.41,
    6.63,   11.38,
    9.42,   5.11,
    16.62,  10.52
};


/* Thresholds */
#define TH_PROB                     (0.3f)
#define TH_NMS                      (0.2f)
/* Size of input image to the model */
#define MODEL_IN_W                  (416)
#define MODEL_IN_H                  (416)


/*****************************************
 * Macro for  ResNet-18 pre YOLOV3
 ******************************************/
/* Number of class to be detected */
/* Model Binary */
const static std::string model_dir1 = "human_gaze_resnet18";
/* Pre-processing Runtime Object */
const static std::string pre_dir1 = model_dir1 + "/preprocess";
/*ResNet-18 Related*/
#define INF_OUT_SIZE_RESNET         (2)
/*Graphic Drawing Settings Related*/
#define KEY_POINT_SIZE              (2)

/*DRP-AI Input image information*/
#define IMAGE_WIDTH                 (640)
#define IMAGE_HEIGHT                (480)
#define DRPAI_IN_WIDTH              (IMAGE_WIDTH)
#define DRPAI_IN_HEIGHT             (IMAGE_HEIGHT)
#define BGRA_CHANNEL                (4)
#define DISP_OUTPUT_WIDTH           (1920)
#define DISP_OUTPUT_HEIGHT          (1080)
#define NUM_MAX_FACE (3)
#define DISP_INF_WIDTH              (1280)
#define DISP_INF_HEIGHT             (960)
/*Total Display out*/

#define DISP_RESIZE_WIDTH            (1550)
#define DISP_RESIZE_HEIGHT           (1080)

#define CLASS_LABEL_HEIGHT           (10)
#define CLASS_LABEL_WIDTH           (90)
#define HEAD_COUNT_STR_X            (645)
#define HEAD_COUNT_STR_Y            (30)

#define T_TIME_STR_X                (645)
#define T_TIME_STR_Y                (120)

#define MODEL_NAME_1_X              (645)
#define MODEL_NAME_1_Y              (190)

#define PRE_TIME_STR_X              (645)
#define PRE_TIME_STR_Y              (240)
#define I_TIME_STR_X                (645)
#define I_TIME_STR_Y                (290)
#define P_TIME_STR_X                (645)
#define P_TIME_STR_Y                (340)

#define MODEL_NAME_2_X              (645)
#define MODEL_NAME_2_Y              (410)

#define PRE_TIME_STR_X_GAZE         (645)
#define PRE_TIME_STR_Y_GAZE         (460)
#define I_TIME_STR_X_GAZE           (645)
#define I_TIME_STR_Y_GAZE           (510)
#define P_TIME_STR_X_GAZE           (645)
#define P_TIME_STR_Y_GAZE           (560)

#define FPS_STR_X                   (645)
#define FPS_STR_Y                   (630)

#define CHAR_SCALE_LARGE            (1.6)
#define CHAR_SCALE_SMALL            (1.2)
#define CHAR_SCALE_XS               (0.5)
#define BOX_THICKNESS               (2)
#define BOX_CHAR_THICKNESS          (0.5)
#define HC_CHAR_THICKNESS           (4)
#define FPS_CHAR_THICKNESS          (4)
#define RIGHT_ALIGN_OFFSET          (20)
#define LINE_HEIGHT                 (30) 
#define LINE_HEIGHT_OFFSET          (20) 

#define CLASS_LABEL_HEIGHT          (10)
#define CLASS_LABEL_WIDTH           (90)

/*Waiting Time*/
#define WAIT_TIME                   (1000) /* microseconds */
#define AI_THREAD_TIMEOUT           (20)  /* seconds */
#define KEY_THREAD_TIMEOUT          (5)   /* seconds */
#define CAPTURE_TIMEOUT             (20)  /* seconds */
#define DISPLAY_THREAD_TIMEOUT      (20)  /* seconds */
#define TIME_COEF  


/* DRPAI_FREQ is the   */
/* frequency settings for DRP-AI.        */
/* Basically use the default values      */

#define DRPAI_FREQ                  (2)
/* DRPAI_FREQ can be set from 1 to 127   */
/* 1,2: 1GHz                             */
/* 3: 630MHz                             */
/* 4: 420MHz                             */
/* 5: 315MHz                             */
/* ...                                   */
/* 127: 10MHz                            */
/* Calculation Formula:                  */
/*     1260MHz /(DRPAI_FREQ - 1)         */
/*     (When DRPAI_FREQ = 3 or more.)    */
#endif
