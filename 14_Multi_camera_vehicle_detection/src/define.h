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
* Description  : DRP-AI Multi camera vehicle detection application
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

/*****************************************
* Static Variables for Tinyyolov3
* Following variables need to be changed in order to custormize the AI model
******************************************/
/*****************************************
* Tinyyolov3
******************************************/
/* Model Binary */
const static std::string model_dir = "Multi_camera_vehicle_detection_tinyyolov3";
/* Pre-processing Runtime Object */
const static std::string pre_dir = model_dir + "/preprocess";

/* DRP-AI memory offset for model object file*/
#define DRPAI_MEM_OFFSET            (0)

/*****************************************
* Macro for Tinyyolov3
******************************************/
/* Number of class to be detected */
#define NUM_CLASS                   (10)
/* Number for [region] layer num parameter */
#define NUM_BB                      (3)
/* Number of output layers. This value MUST match with the length of num_grids[] below */
#define NUM_INF_OUT_LAYER           (2)
/* Number of grids in the image. The length of this array MUST match with the NUM_INF_OUT_LAYER */
const static uint8_t num_grids[] = { 13,26 };

/* Number of DRP-AI output */
const static uint32_t INF_OUT_SIZE = (NUM_CLASS + 5) * NUM_BB * num_grids[0] * num_grids[0]
                                   + (NUM_CLASS + 5) * NUM_BB * num_grids[1] * num_grids[1];

/* Anchor box information */
const static double anchors[] =
{
    10, 14,
    23, 27,
    37, 58,
    81, 82,
    135, 169,
    344, 319
};

/* Thresholds */
#define TH_PROB                     (0.5f)
#define TH_NMS                      (0.3f)
/* Size of input image to the model */
#define MODEL_IN_W                  (416)
#define MODEL_IN_H                  (416)

/*DRP-AI Input image information*/
#define DRP_RESIZE_WIDTH            (640)
#define DRP_RESIZE_HEIGHT           (480)
#define IMAGE_WIDTH                 (640)
#define IMAGE_HEIGHT                (480)
#define DRPAI_IN_WIDTH              (IMAGE_WIDTH)
#define DRPAI_IN_HEIGHT             (IMAGE_HEIGHT)
#define BGRA_CHANNEL                (4)
#define BGR_CHANNEL                 (3)
#define DISP_OUTPUT_WIDTH           (1920)
#define DISP_OUTPUT_HEIGHT          (1080)
#define DISP_INF_WIDTH              (1920)
#define DISP_INF_HEIGHT             (1080)

#define DISP_RESIZE_WIDTH            (1550)
#define DISP_RESIZE_HEIGHT           (1080)

/*Image:: Text information to be drawn on image*/
#define BOX_LINE_SIZE               (3)
#define BOX_HEIGHT_OFFSET           (20)
#define BOX_TEXT_HEIGHT_OFFSET      (17)
#define CLASS_LABEL_HEIGHT          (10)
#define CLASS_LABEL_WIDTH           (100)
#define PRE_TIME_STR_Y              (550)
#define P_TIME_STR_X                (645)
#define APP_NAME_X                  (0)
#define P_TIME_STR_Y                (650)
#define I_TIME_STR_X                (645)
#define I_TIME_STR_Y                (600)
#define T_TIME_STR_Y                (500)
#define CHAR_SCALE_LARGE            (1.6)
#define CHAR_SCALE_SMALL            (1.2)
#define CHAR_SCALE_APP_NAME         (1.0)
#define CHAR_SCALE_XS               (0.5)
#define BOX_THICKNESS               (2)
#define BOX_CHAR_THICKNESS          (0.5)
#define HC_CHAR_THICKNESS           (4)
#define FPS_CHAR_THICKNESS          (4)
#define RIGHT_ALIGN_OFFSET          (20)
#define HEIGHT_OFFSET               (20)

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
