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
* Description  : DRP-AI TVM[*1] Application for Road Lane Segmentation
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
* Static Variables for Unet
* Following variables need to be changed in order to custormize the AI model
******************************************/
/*****************************************
* Unet
******************************************/
/* Model Binary */
const static std::string model_dir = "unet_onnx";

/* DRP-AI memory offset for model object file*/
#define DRPAI_MEM_OFFSET            (0)

/* Thresholds */
#define TH_PROB                     (0.01f)

/*Model input info*/
#define MODEL_IN_H                  (224)
#define MODEL_IN_W                  (224)

/*DRP-AI Input image information*/
#define IMAGE_WIDTH                 (640)
#define IMAGE_HEIGHT                (480)
#define DISP_OUTPUT_WIDTH           (1920)
#define DISP_OUTPUT_HEIGHT          (1080)
#define DISP_INF_WIDTH              (1280)
#define DISP_INF_HEIGHT             (960)
#define BGR_CHANNEL                 (3)
#define BGRA_CHANNEL                (4)

/*Image:: Text information to be drawn on image*/
#define FPS_STR_X                   (645)
#define FPS_STR_Y                   (360)
#define PRE_TIME_STR_X              (645)
#define PRE_TIME_STR_Y              (170)
#define P_TIME_STR_X                (645)
#define P_TIME_STR_Y                (270)
#define I_TIME_STR_X                (645)
#define I_TIME_STR_Y                (220)
#define T_TIME_STR_X                (645)
#define T_TIME_STR_Y                (120)
#define TEMP_STR_X                  (645)
#define TEMP_STR_Y                  (960)
#define CHAR_SCALE_LARGE            (1.6)
#define CHAR_SCALE_SMALL            (1.2)
#define CHAR_SCALE_XS               (0.5)
#define BOX_THICKNESS               (2)
#define BOX_CHAR_THICKNESS          (0.5)
#define LANE_CHAR_THICKNESS         (4)
#define FPS_CHAR_THICKNESS          (4)
#define RIGHT_ALIGN_OFFSET          (20)

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