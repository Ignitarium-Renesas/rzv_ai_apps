/*
 * Original Code (C) Copyright Renesas Electronics Corporation 2023
 * Modified Code (C) Copyright Renesas Electronics Corporation 2024
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
* Version      : 1.0.0
* Description  : DRP-AI Driver monitoring system detection application
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
#include <numeric>


#define MIPI_CAM_RES "640x480"

/*****************************************
* Static Variables for yolox
* Following variables need to be changed in order to custormize the AI model
******************************************/
/*****************************************
* yolox
******************************************/
/* Model Binary */
const static std::string model_dir = "dms_yolox-l_onnx";
/* Pre-processing Runtime Object */
const static std::string pre_dir = model_dir + "/preprocess";

/* DRP-AI memory offset for model object file*/
#define DRPAI_MEM_OFFSET            (0)

/*****************************************
* Macro for yolox
******************************************/
/* Number of class to be detected */
#define NUM_CLASS                   (10)
/* Number for [region] layer num parameter */
#define NUM_BB                      (1)
/* Number of output layers. This value MUST match with the length of num_grids[] below */
#define NUM_INF_OUT_LAYER           (3)
/* Number of grids in the image. The length of this array MUST match with the NUM_INF_OUT_LAYER */
const static uint8_t num_grids[] = { 36,18,9 };

/* Number of DRP-AI output */
const static uint32_t INF_OUT_SIZE = (NUM_CLASS + 5) * NUM_BB * num_grids[0] * num_grids[0]
                                   + (NUM_CLASS + 5) * NUM_BB * num_grids[1] * num_grids[1]
                                   + (NUM_CLASS + 5) * NUM_BB * num_grids[2] * num_grids[2];


/* Thresholds */
#define TH_PROB                     (0.5f)
#define TH_NMS                      (0.3f)
/* Size of input image to the model */
#define MODEL_IN_W                  (288)
#define MODEL_IN_H                  (288)

/*DRP-AI Input image information*/
#define IMAGE_WIDTH                 (640)
#define IMAGE_HEIGHT                (480)
#define DRPAI_IN_WIDTH              (IMAGE_WIDTH)
#define DRPAI_IN_HEIGHT             (IMAGE_HEIGHT)
#define BGRA_CHANNEL                (4)
#define BGR_CHANNEL                 (3)
#define DISP_OUTPUT_WIDTH           (1920)
#define DISP_OUTPUT_HEIGHT          (1080)
#define DISP_INF_WIDTH              (1280)
#define DISP_INF_HEIGHT             (960)

/*Image:: Text information to be drawn on image*/
#define CHAR_SCALE_LARGE            (1.6)
#define CHAR_SCALE_SMALL            (1.2)
#define FIRST_FRAME_X_COORDINATE    (0)
#define FIRST_FRAME_Y_COORDINATE    (60)

/*Total Display out*/

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

#define FPS_STR_X                   (645)
#define FPS_STR_Y                   (630)

#define HEAD_POSE_STR_X             (20)
#define HEAD_POSE_STR_Y             (20)
#define BLINK_STR_X                 (20)
#define BLINK_STR_Y                 (40)
#define YAWN_STR_X                  (20)
#define YAWN_STR_Y                  (60)

#define CHAR_SCALE_LARGE            (1.6)
#define CHAR_SCALE_SMALL            (1.2)
#define DMS_CHAR_SCALE_SMALL        (0.5)
#define HC_CHAR_THICKNESS           (4)
#define DMS_CHAR_THICKNESS          (1.9)
#define RIGHT_ALIGN_OFFSET          (20)

/*Waiting Time*/
#define WAIT_TIME                   (1000) /* microseconds */
#define AI_THREAD_TIMEOUT           (20)  /* seconds */
#define KEY_THREAD_TIMEOUT          (5)   /* seconds */
#define CAPTURE_TIMEOUT             (20)  /* seconds */
#define DISPLAY_THREAD_TIMEOUT      (20)  /* seconds */

/* OpenCVA Circuit Number */
#define DRP_FUNC_NUM            (16)
#define DRP_FUNC_RESIZE         (0)
#define DRP_FUNC_CVT_YUV2BGR    (2)
#define DRP_FUNC_CVT_NV2BGR     (2)
#define DRP_FUNC_GAUSSIAN       (4)
#define DRP_FUNC_DILATE         (5)
#define DRP_FUNC_ERODE          (6)
#define DRP_FUNC_FILTER2D       (7)
#define DRP_FUNC_SOBEL          (8)
#define DRP_FUNC_A_THRESHOLD    (9)
#define DRP_FUNC_TMPLEATMATCH   (10)
#define DRP_FUNC_AFFINE         (11)
#define DRP_FUNC_PYR_DOWN       (12)
#define DRP_FUNC_PYR_UP         (13)
#define DRP_FUNC_PERSPECTIVE    (14)

/* OpenCVA Activate */
#define OPENCVA_FUNC_DISABLE    (0)
#define OPENCVA_FUNC_ENABLE     (1)

/* DRP_MAX_FREQ and DRPAI_FREQ are the   */
/* frequency settings for DRP-AI.        */
/* Basically use the default values      */

#define DRPAI_FREQ              (2)
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
