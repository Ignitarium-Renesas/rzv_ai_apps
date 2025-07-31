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
* Description  : DRP-AI TVM[*1] Application for Elderly Fall Detection

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
* Static Variables for TinyYOLOv2
* Following variables need to be changed in order to custormize the AI model
******************************************/
/*****************************************
* TinyYOLOv2
******************************************/
/* Model Binary */
const static std::string model_dir = "elderly_fall_detection_tinyyolov2";

/* Pre-processing Runtime Object */
const static std::string pre_dir = model_dir + "/preprocess";

/* Label list file */
const static std::string label_list = "labels.txt";

/* Empty List to store label list */
static std::vector<std::string> label_file_map = {};

/* DRP-AI memory offset for model object file*/
#define DRPAI_MEM_OFFSET              (0x0000000)
#define DRPAI_MEM_OFFSET1             (0x2000000)

/*****************************************
 * Macro for TinyYOLOv2
 ******************************************/
/* Number of class to be detected */
#define NUM_CLASS                   (1)

/* Number for [region] layer num parameter */
#define NUM_BB                      (5)
#define NUM_GRID_X                  (13)
#define NUM_GRID_Y                  (13)
const static uint8_t num_grids[] = { 13 };
const static uint32_t INF_OUT_SIZE_TINYYOLOV2 = (NUM_CLASS + 5)* NUM_BB * num_grids[0] * num_grids[0];
const static double anchors[] =
{
    1.08,   1.19,
    3.42,   4.41,
    6.63,   11.38,
    9.42,   5.11,
    16.62,  10.52
};

/* Thresholds */
#define TH_PROB                     (0.6f)
#define TH_NMS                      (0.3f)

/* Size of input image to the model */
#define MODEL_IN_W                  (416)
#define MODEL_IN_H                  (416)

/*****************************************
 * Macro for  HRNet-18 pre TinyYOLOv2
 ******************************************/
/* Model Binary */
const static std::string model_dir1 = "elderly_fall_detection_hrnet";

/* Pre-processing Runtime Object */
const static std::string pre_dir1 = model_dir1 + "/preprocess";

/*HRNet Related*/
#define NUM_OUTPUT_W                (64)
#define NUM_OUTPUT_H                (64)
#define NUM_OUTPUT_C                (16)
#define INF_OUT_SIZE_HRNET          (NUM_OUTPUT_W*NUM_OUTPUT_H*NUM_OUTPUT_C)

/* Size of input image to the model */
#define MODEL_IN_W_HRNET            (256)
#define MODEL_IN_H_HRNET            (256)

/*HRNet Post Processing & Drawing Related*/
#define TH_KPT                      (0.001f)
#define OUTPUT_ADJ_X                (2)
#define OUTPUT_ADJ_Y                (0)
#define NUM_MAX_PERSON              (3)
#define CROP_ADJ_X                  (20)
#define CROP_ADJ_Y                  (20)
#define KEY_POINT_SIZE              (4)

/*DRP-AI Input image information*/
#define IMAGE_WIDTH                 (640)
#define IMAGE_HEIGHT                (480)
#define DRPAI_IN_WIDTH              (IMAGE_WIDTH)
#define DRPAI_IN_HEIGHT             (IMAGE_HEIGHT)
#define BGRA_CHANNEL                (4)
#define DISP_OUTPUT_WIDTH           (1920)
#define DISP_OUTPUT_HEIGHT          (1080)
#define DISP_INF_WIDTH              (1280)
#define DISP_INF_HEIGHT             (960)

/*Image:: Text information to be drawn on image*/         
#define MODEL_NAME_1_Y              (190)
#define MODEL_NAME_2_Y              (410) 
#define T_TIME_STR_Y                (120)         
#define PRE_TIME_STR_Y              (240)         
#define I_TIME_STR_Y                (290)          
#define P_TIME_STR_Y                (340)
#define PRE_TIME_STR_Y_HRNET        (460)
#define I_TIME_STR_Y_HRNET          (510)
#define P_TIME_STR_Y_HRNET          (560)        
#define FPS_STR_Y                   (630)        
#define CHAR_SCALE_LARGE            (1.6)
#define CHAR_SCALE_SMALL            (1.2)
#define HC_CHAR_THICKNESS           (4)
#define RIGHT_ALIGN_OFFSET          (20)
#define PERSON_STR_X                (130)
#define PERSON_STR_Y                (240)
#define PERSON_CHAR_THICKNESS       (2.5)
#define PERSON_SCALE_SMALL          (1) 

/*Waiting Time*/
#define WAIT_TIME                   (1000) /* microseconds */
#define AI_THREAD_TIMEOUT           (20)  /* seconds */
#define KEY_THREAD_TIMEOUT          (5)   /* seconds */
#define CAPTURE_TIMEOUT             (20)  /* seconds */
#define DISPLAY_THREAD_TIMEOUT      (20)  /* seconds */
#define TIME_COEF  

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
