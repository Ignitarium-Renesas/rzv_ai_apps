/*
 * Original Code (C) Copyright Edgecortix, Inc. 2022
 * Modified Code (C) Copyright Renesas Electronics Corporation 2023-2024
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
* File Name    : multi_camera_vehicle_detection.cpp
* Version      : 1.0.0
* Description  : DRP-AI Multi camera vehicle detection application
***********************************************************************************************************************/

/*****************************************
* includes
******************************************/
#include "define.h"
#include "define_color.h"
#include "box.h"
#include "MeraDrpRuntimeWrapper.h"
#include <linux/drpai.h>
#include <linux/input.h>
#include <builtin_fp16.h>
#include <opencv2/opencv.hpp>
#include "wayland.h"

/*Pre-processing Runtime Header*/
#include "PreRuntime.h"
/*dmabuf for Pre-processing Runtime input data*/
#include "dmabuf.h"

using namespace std;
using namespace cv;

/* DRP-AI TVM[*1] Runtime object */
MeraDrpRuntimeWrapper runtime;
/* Pre-processing Runtime object */
PreRuntime preruntime;

/*MMNGR buffer for DRP-AI Pre-processing*/
static dma_buffer *drpai_buf;
static dma_buffer *drpai_buf1;
static dma_buffer *drpai_buf2;
static dma_buffer *drpai_buf3;

/*Global Variables*/
static float drpai_output_buf[INF_OUT_SIZE];
static float drpai_output_buf1[INF_OUT_SIZE];
static float drpai_output_buf2[INF_OUT_SIZE];
static float drpai_output_buf3[INF_OUT_SIZE];
static Wayland wayland;
static pthread_t ai_inf_thread;
static pthread_t kbhit_thread;
static pthread_t capture_thread;
static pthread_t memcpy_thread;
static mutex mtx;
static mutex mtx1;

/*Flags*/
static std::atomic<uint8_t> capture_start           (1);
static std::atomic<uint8_t> inference_start         (0);
static std::atomic<uint8_t> img_processing_start    (0);
float sum_inf_time;
float sum_pre_poc_time;
float sum_post_proc_time;
float sum_total_time;

static sem_t terminate_req_sem;
static int32_t drpai_freq;
static int32_t flip_mode=0;
static int32_t number_of_cameras;
std::vector<std::string> gstreamer_pipeline;
std::vector<std::string> device_paths;
std::vector<VideoCapture> cap;

uint64_t drpaimem_addr_start = 0;
bool runtime_status = false; 
static vector<detection> det;
static vector<detection> det1;
static vector<detection> det2;
static vector<detection> det3;

/*Global frame */
std::vector<Mat> g_frame;
std::vector<Mat> cap_frame;
std::vector<Mat> inf_frames=
{
   cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0)),
   cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0)),
   cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0)),
   cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0))
};
std::vector<Mat> img_frames=
{
    cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0)),
    cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0)),
    cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0)),
    cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0))
};

/* Map to store input source list */
std::map<std::string, int> input_source_map ={    
    {"USB", 1},
    {"MIPI", 2}
    } ;


/*****************************************
 * Function Name     : float16_to_float32
 * Description       : Function by Edgecortex. Cast uint16_t a into float value.
 * Arguments         : a = uint16_t number
 * Return value      : float = float32 number
 ******************************************/
float float16_to_float32(uint16_t a)
{
    return __extendXfYf2__<uint16_t, uint16_t, 10, float, uint32_t, 23>(a);
}


/*****************************************
* Function Name : sigmoid
* Description   : Helper function for YOLO Post Processing
* Arguments     : x = input argument for the calculation
* Return value  : sigmoid result of input x
******************************************/
double sigmoid(double x)
{
    return 1.0/(1.0 + exp(-x));
}

/*****************************************
* Function Name : yolo_index
* Description   : Get the index of the bounding box attributes based on the input offset
* Arguments     : n = output layer number
                  offs = offset to access the bounding box attributes
*                 channel = channel to access each bounding box attribute.
* Return value  : index to access the bounding box attribute.
******************************************/
int32_t yolo_index(uint8_t n, int32_t offs, int32_t channel)
{
    uint8_t num_grid = num_grids[n];
    return offs + channel * num_grid * num_grid;
}

/*****************************************
* Function Name : yolo_offset
* Description   : Get the offset number to access the bounding box attributes
*                 To get the actual value of bounding box attributes, use yolo_index() after this function.
* Arguments     : n = output layer number [0~2].
                  b = Number to indicate which bounding box in the region [0~4]
*                 y = Number to indicate which region [0~13]
*                 x = Number to indicate which region [0~13]
* Return value  : offset to access the bounding box attributes.
******************************************/
int32_t yolo_offset(uint8_t n, int32_t b, int32_t y, int32_t x)
{
    uint8_t num = num_grids[n];
    uint32_t prev_layer_num = 0;
    int32_t i = 0;

    for (i = 0; i < n; i++)
    {
        prev_layer_num += NUM_BB * (NUM_CLASS + 5) * num_grids[i] * num_grids[i];
    }
    return prev_layer_num + b * (NUM_CLASS + 5) * num * num + y * num + x;
}

static int8_t wait_join(pthread_t *p_join_thread, uint32_t join_time)
{
    int8_t ret_err;
    struct timespec join_timeout;
    ret_err = clock_gettime(CLOCK_REALTIME, &join_timeout);
    if ( 0 == ret_err )
    {
        join_timeout.tv_sec += join_time;
        ret_err = pthread_timedjoin_np(*p_join_thread, NULL, &join_timeout);
    }
    return ret_err;
}

/*****************************************
* Function Name : R_Post_Proc
* Description   : Process CPU post-processing for tinyyolov3
* Arguments     : floatarr = drpai output address
                  camera_num = index of camera
* Return value  : -
******************************************/
void R_Post_Proc(float* floatarr, int camera_num)
{
    /* Following variables are required for correct_region_boxes in Darknet implementation*/
    /* Note: This implementation refers to the "darknet detector test" */
    vector<detection> det_buff;
    float new_w, new_h;
    float correct_w = 1.;
    float correct_h = 1.;
    if ((float) (MODEL_IN_W / correct_w) < (float) (MODEL_IN_H/correct_h) )
    {
        new_w = (float) MODEL_IN_W;
        new_h = correct_h * MODEL_IN_W / correct_w;
    }
    else
    {
        new_w = correct_w * MODEL_IN_H / correct_h;
        new_h = MODEL_IN_H;
    }

    int32_t n = 0;
    int32_t b = 0;
    int32_t y = 0;
    int32_t x = 0;
    int32_t offs = 0;
    int32_t i = 0;
    float tx = 0;
    float ty = 0;
    float tw = 0;
    float th = 0;
    float tc = 0;
    float center_x = 0;
    float center_y = 0;
    float box_w = 0;
    float box_h = 0;
    float objectness = 0;
    uint8_t num_grid = 0;
    uint8_t anchor_offset = 0;
    float classes[NUM_CLASS];
    float max_pred = 0;
    int32_t pred_class = -1;
    float probability = 0;
    detection d;
    //YOLOX-L
    int stride = 0;
    vector<int> strides = {8, 16, 32};

    for (n = 0; n < NUM_INF_OUT_LAYER; n++)
    {
        num_grid = num_grids[n];
        anchor_offset = 2 * NUM_BB * (NUM_INF_OUT_LAYER - (n + 1));

        for (b = 0; b < NUM_BB; b++)
        {
           stride = strides[n];
            for (y = 0;y<num_grid;y++)
            {
                for (x = 0;x<num_grid;x++)
                {
                    offs = yolo_offset(n, b, y, x);
                    tc = floatarr[yolo_index(n, offs, 4)];

                    objectness = tc;

                    if (objectness > TH_PROB)
                    {
                        /* Get the class prediction */
                        for (i = 0; i < NUM_CLASS; i++)
                        {
                            classes[i] = floatarr[yolo_index(n, offs, 5+i)];
                        }

                        max_pred = 0;
                        pred_class = -1;
                        for (i = 0; i < NUM_CLASS; i++)
                        {
                            if (classes[i] > max_pred)
                            {
                                pred_class = i;
                                max_pred = classes[i];
                            }
                        }

                        /* Store the result into the list if the probability is more than the threshold */
                        probability = max_pred * objectness;
                        if (probability > TH_PROB)
                        {
                            tx = floatarr[offs];
                            ty = floatarr[yolo_index(n, offs, 1)];
                            tw = floatarr[yolo_index(n, offs, 2)];
                            th = floatarr[yolo_index(n, offs, 3)];

                            /* Compute the bounding box */
                            /*get_yolo_box/get_region_box in paper implementation*/
                            center_x = (tx+ float(x))* stride;
                            center_y = (ty+ float(y))* stride;
                            center_x = center_x  / (float) MODEL_IN_W;
                            center_y = center_y  / (float) MODEL_IN_H;
                            box_w = exp(tw) * stride;
                            box_h = exp(th) * stride;
                            box_w = box_w / (float) MODEL_IN_W;
                            box_h = box_h / (float) MODEL_IN_H;

                            /* Adjustment for size */
                            /* correct_yolo/region_boxes */
                            center_x = (center_x - (MODEL_IN_W - new_w) / 2. / MODEL_IN_W) / ((float) new_w / MODEL_IN_W);
                            center_y = (center_y - (MODEL_IN_H - new_h) / 2. / MODEL_IN_H) / ((float) new_h / MODEL_IN_H);
                            box_w *= (float) (MODEL_IN_W / new_w);
                            box_h *= (float) (MODEL_IN_H / new_h);

                            center_x = round(center_x * DRPAI_IN_WIDTH);
                            center_y = round(center_y * DRPAI_IN_HEIGHT);
                            box_w = round(box_w * DRPAI_IN_WIDTH);
                            box_h = round(box_h * DRPAI_IN_HEIGHT);

                            Box bb = {center_x, center_y, box_w, box_h};
                            d = {bb, pred_class, probability};
                            det_buff.push_back(d);
                        }
                    }
                }
            }
        }
    }
    /* Non-Maximum Supression filter */
    filter_boxes_nms(det_buff, det_buff.size(), TH_NMS);

    mtx.lock();
    /* Clear the detected result list */
    if(camera_num == 0)
    {
        det.clear();
        copy(det_buff.begin(), det_buff.end(), back_inserter(det));
    }
    else if(camera_num == 1)
    {
        det1.clear();
        copy(det_buff.begin(), det_buff.end(), back_inserter(det1));
    }
    else if(camera_num == 2)
    {
        det2.clear();
        copy(det_buff.begin(), det_buff.end(), back_inserter(det2));
    }
    else if(camera_num == 3)
    {
        det3.clear();
        copy(det_buff.begin(), det_buff.end(), back_inserter(det3));
    }
    mtx.unlock();
    return ;
}

/*****************************************
 * Function Name : draw_bounding_box
 * Description   : Draw bounding box on image.
 * Arguments     : -
 * Return value  : -
 ******************************************/
void draw_bounding_box(void)
{
    vector<detection> det_buff;
    vector<detection> det_buff1;
    vector<detection> det_buff2;
    vector<detection> det_buff3;
    stringstream stream;
    string result_str;
    int32_t i = 0;
    uint32_t color=0;

    vector<detection> tmp_buff;

    int32_t cam_num;
    cv::Mat tmp_image;

    int baseline = 0;

    for(cam_num = 0; cam_num < number_of_cameras; cam_num++)
    {
             if(cam_num == 0){tmp_buff = det_buff;  tmp_image = img_frames[cam_num];}
        else if(cam_num == 1){tmp_buff = det_buff1; tmp_image = img_frames[cam_num];}
        else if(cam_num == 2){tmp_buff = det_buff2; tmp_image = img_frames[cam_num];}
        else if(cam_num == 3){tmp_buff = det_buff3; tmp_image = img_frames[cam_num];}

        mtx.lock();
             if(cam_num == 0){ copy(det.begin(),  det.end(),  back_inserter(tmp_buff)); }
        else if(cam_num == 1){ copy(det1.begin(), det1.end(), back_inserter(tmp_buff)); }
        else if(cam_num == 2){ copy(det2.begin(), det2.end(), back_inserter(tmp_buff)); }
        else if(cam_num == 3){ copy(det3.begin(), det3.end(), back_inserter(tmp_buff)); }
        mtx.unlock();

        /* Draw bounding box on RGB image. */
        for (i = 0; i < tmp_buff.size(); i++)
        {
            /* Skip the overlapped bounding boxes */
            if (tmp_buff[i].prob == 0) continue;

            color = box_color[tmp_buff[i].c];
            /* Clear string stream for bounding box labels */
            stream.str("");
            /* Draw the bounding box on the image */
            stream << fixed << setprecision(2) << tmp_buff[i].prob;
            result_str = label_file_map[tmp_buff[i].c]+ " "+ stream.str();

            int32_t x_min = (int)tmp_buff[i].bbox.x - round((int)tmp_buff[i].bbox.w / 2.);
            int32_t y_min = (int)tmp_buff[i].bbox.y - round((int)tmp_buff[i].bbox.h / 2.);
            int32_t x_max = (int)tmp_buff[i].bbox.x + round((int)tmp_buff[i].bbox.w / 2.) - 1;
            int32_t y_max = (int)tmp_buff[i].bbox.y + round((int)tmp_buff[i].bbox.h / 2.) - 1;

            /* Check the bounding box is in the image range */
            x_min = x_min < 1 ? 1 : x_min;
            x_max = ((DRPAI_IN_WIDTH - 2) < x_max) ? (DRPAI_IN_WIDTH - 2) : x_max;
            y_min = y_min < 1 ? 1 : y_min;
            y_max = ((DRPAI_IN_HEIGHT - 2) < y_max) ? (DRPAI_IN_HEIGHT - 2) : y_max;

            uint8_t r = (color >> 16) & 0x0000FF;
            uint8_t g = (color >>  8) & 0x0000FF;
            uint8_t b = color & 0x0000FF;

            cv::rectangle(tmp_image, cv::Point(x_min,y_min), cv::Point(x_max,y_max), cv::Scalar(b, g, r, 0xFF), 2);

            cv::Size size = cv::getTextSize(result_str.c_str(), cv::FONT_HERSHEY_TRIPLEX,CLASS_NAME_FONT ,CLASS_NAME_THICKNESS, &baseline);
            cv::rectangle(tmp_image, cv::Point(x_min+2, y_min+2), cv::Point(x_min+2+size.width,y_min+2+size.height+2), cv::Scalar(0,0,0), cv::FILLED);
            /*Color must be in BGR order*/
            cv::putText(tmp_image, result_str.c_str(), cv::Point(x_min+2, y_min+2+size.height), cv::FONT_HERSHEY_TRIPLEX, CLASS_NAME_FONT, cv::Scalar(255, 255, 255),CLASS_NAME_THICKNESS);
        }
    }
    return;
}

/*****************************************
 * Function Name : Vehicle Detection
 * Description   : Function to perform over all detection
 * Arguments     : -
 * Return value  : 0 if succeeded
 *                 not 0 otherwise
 ******************************************/
int Vehicle_Detection()
{ 
    
    /*Variable for getting Inference output data*/
    void* output_ptr;
    void* output_ptr1;
    void* output_ptr2;
    void* output_ptr3;

    uint32_t out_size;
    uint32_t out_size1;
    uint32_t out_size2;
    uint32_t out_size3;

    /*Variable for Pre-processing parameter configuration*/
    s_preproc_param_t in_param;

    /*Variable for checking return value*/
    int8_t ret = 0;

    /*load inference out on drpai_out_buffer*/
    int32_t i = 0;

    int32_t inf_num;

    int32_t output_num = 0;
    int32_t output_num1 = 0;
    int32_t output_num2 = 0;
    int32_t output_num3 = 0;

    std::tuple<InOutDataType, void *, int64_t> output_buffer;
    std::tuple<InOutDataType, void *, int64_t> output_buffer1;
    std::tuple<InOutDataType, void *, int64_t> output_buffer2;
    std::tuple<InOutDataType, void *, int64_t> output_buffer3;

    int64_t output_size;
    int64_t output_size1;
    int64_t output_size2;
    int64_t output_size3;

    uint32_t size_count  = 0;
    uint32_t size_count1 = 0;
    uint32_t size_count2 = 0;
    uint32_t size_count3 = 0;
    
    mtx1.lock();
    sum_inf_time=0;
    sum_pre_poc_time=0;
    sum_post_proc_time=0;
    sum_total_time=0;

    for(inf_num=0; inf_num<number_of_cameras; inf_num++)
    {
        /* camera0 */
        if(inf_num==0)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            in_param.pre_in_shape_w = IMAGE_WIDTH;
            in_param.pre_in_shape_h = IMAGE_HEIGHT;

            in_param.pre_in_addr = (uintptr_t) drpai_buf->phy_addr;

            ret = preruntime.Pre(&in_param, &output_ptr, &out_size);
            if (0 < ret)
            {
                fprintf(stderr, "[ERROR] Failed to run Pre-processing Runtime Pre0()\n");
                return 0;
            }

            auto t1 = std::chrono::high_resolution_clock::now();

            auto pre_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

            /*Set Pre-processing output to be inference input. */
            runtime.SetInput(0, (float*)output_ptr);
            auto t2 = std::chrono::high_resolution_clock::now();
            runtime.Run(drpai_freq);
            auto t3 = std::chrono::high_resolution_clock::now();
            auto inf_duration = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
            auto t4 = std::chrono::high_resolution_clock::now();
            /* Get the number of output of the target model. */
            output_num = runtime.GetNumOutput();

            size_count = 0;
            /*GetOutput loop*/
            for (i = 0; i < output_num; i++)
            {
                /* output_buffer below is tuple, which is { data type, address of output data, number of elements } */
                output_buffer = runtime.GetOutput(i);
                /*Output Data Size = std::get<2>(output_buffer). */
                output_size = std::get<2>(output_buffer);

                /*Output Data Type = std::get<0>(output_buffer)*/
                if (InOutDataType::FLOAT16 == std::get<0>(output_buffer))
                {
                    /*Output Data = std::get<1>(output_buffer)*/
                    uint16_t *data_ptr = reinterpret_cast<uint16_t *>(std::get<1>(output_buffer));
                    for (int j = 0; j < output_size; j++)
                    {
                        /*FP16 to FP32 conversion*/
                        drpai_output_buf[j + size_count] = float16_to_float32(data_ptr[j]);
                    }
                }
                else if (InOutDataType::FLOAT32 == std::get<0>(output_buffer))
                {
                    /*Output Data = std::get<1>(output_buffer)*/
                    float *data_ptr = reinterpret_cast<float *>(std::get<1>(output_buffer));
                    for (int j = 0; j < output_size; j++)
                    {
                        drpai_output_buf[j + size_count] = data_ptr[j];
                    }
                }
                else
                {
                    std::cerr << "[ERROR] Output data type : not floating point." << std::endl;
                    ret = -1;
                    break;
                }
                size_count += output_size;
            }
            if (ret != 0)
            {
                std::cerr << "[ERROR] DRP Inference Not working !!! " << std::endl;
                return -1;
            }

            R_Post_Proc(drpai_output_buf,  0);
            auto t5 = std::chrono::high_resolution_clock::now();
            
            auto r_post_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
            float total_time = float(inf_duration/1000.0) + float(r_post_proc_time/1000.0) + float(pre_proc_time/1000.0); 
            sum_inf_time+=inf_duration/1000.0;
            sum_pre_poc_time+=pre_proc_time/1000.0;
            sum_post_proc_time+=r_post_proc_time/1000.0;
            sum_total_time+=total_time;
        }
        /* camera1 */
        else if(inf_num==1)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            
	        in_param.pre_in_shape_w = IMAGE_WIDTH;
            in_param.pre_in_shape_h = IMAGE_HEIGHT;

            in_param.pre_in_addr = (uintptr_t) drpai_buf1->phy_addr;

            ret = preruntime.Pre(&in_param, &output_ptr1, &out_size1);
            if (0 < ret)
            {
                fprintf(stderr, "[ERROR] Failed to run Pre-processing Runtime Pre1()\n");
                return 0;
            }

            auto t1 = std::chrono::high_resolution_clock::now();

            auto pre_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            /*Set Pre-processing output to be inference input. */
            runtime.SetInput(0, (float*)output_ptr1);
            auto t2 = std::chrono::high_resolution_clock::now();
            runtime.Run(drpai_freq);
            auto t3 = std::chrono::high_resolution_clock::now();
            auto inf_duration = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
            auto t4 = std::chrono::high_resolution_clock::now();
            /* Get the number of output of the target model. */
            output_num1 = runtime.GetNumOutput();

            size_count1 = 0;
            /*GetOutput loop*/
            for (i = 0; i < output_num1; i++)
            {
                /* output_buffer below is tuple, which is { data type, address of output data, number of elements } */
                output_buffer1 = runtime.GetOutput(i);
                /*Output Data Size = std::get<2>(output_buffer). */
                output_size1 = std::get<2>(output_buffer1);

                /*Output Data Type = std::get<0>(output_buffer)*/
                if (InOutDataType::FLOAT16 == std::get<0>(output_buffer1))
                {
                    /*Output Data = std::get<1>(output_buffer)*/
                    uint16_t *data_ptr1 = reinterpret_cast<uint16_t *>(std::get<1>(output_buffer1));
                    for (int j = 0; j < output_size1; j++)
                    {
                        /*FP16 to FP32 conversion*/
                        drpai_output_buf1[j + size_count1] = float16_to_float32(data_ptr1[j]);
                    }
                }
                else if (InOutDataType::FLOAT32 == std::get<0>(output_buffer1))
                {
                    /*Output Data = std::get<1>(output_buffer)*/
                    float *data_ptr1 = reinterpret_cast<float *>(std::get<1>(output_buffer1));
                    for (int j = 0; j < output_size1; j++)
                    {
                        drpai_output_buf1[j + size_count1] = data_ptr1[j];
                    }
                }
                else
                {
                    std::cerr << "[ERROR] Output data type : not floating point." << std::endl;
                    ret = -1;
                    break;
                }
                size_count1 += output_size1;
            }
            if (ret != 0)
            {
                std::cerr << "[ERROR] DRP Inference Not working !!! " << std::endl;
                return -1;
            }

            R_Post_Proc(drpai_output_buf1, 1);
            auto t5 = std::chrono::high_resolution_clock::now();
            
            auto r_post_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
            float total_time = float(inf_duration/1000.0) + float(r_post_proc_time/1000.0) + float(pre_proc_time/1000.0); 
            sum_inf_time+=inf_duration/1000.0;
            sum_pre_poc_time+=pre_proc_time/1000.0;
            sum_post_proc_time+=r_post_proc_time/1000.0;
            sum_total_time+=total_time;
        }
        /* camera2 */
        else if(inf_num==2)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
	        in_param.pre_in_shape_w = IMAGE_WIDTH;
            in_param.pre_in_shape_h = IMAGE_HEIGHT;

            in_param.pre_in_addr = (uintptr_t) drpai_buf2->phy_addr;

            ret = preruntime.Pre(&in_param, &output_ptr2, &out_size2);
            if (0 < ret)
            {
                fprintf(stderr, "[ERROR] Failed to run Pre-processing Runtime Pre2()\n");
                return 0;
            }

            auto t1 = std::chrono::high_resolution_clock::now();

            auto pre_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

            /*Set Pre-processing output to be inference input. */
            runtime.SetInput(0, (float*)output_ptr2);
            auto t2 = std::chrono::high_resolution_clock::now();
            runtime.Run(drpai_freq);
            auto t3 = std::chrono::high_resolution_clock::now();
            auto inf_duration = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
            auto t4 = std::chrono::high_resolution_clock::now();
            /* Get the number of output of the target model. */
            output_num2 = runtime.GetNumOutput();

            size_count2 = 0;
            /*GetOutput loop*/
            for (i = 0; i < output_num2; i++)
            {
                /* output_buffer below is tuple, which is { data type, address of output data, number of elements } */
                output_buffer2 = runtime.GetOutput(i);
                /*Output Data Size = std::get<2>(output_buffer). */
                output_size2 = std::get<2>(output_buffer2);

                /*Output Data Type = std::get<0>(output_buffer)*/
                if (InOutDataType::FLOAT16 == std::get<0>(output_buffer2))
                {
                    /*Output Data = std::get<1>(output_buffer)*/
                    uint16_t *data_ptr2 = reinterpret_cast<uint16_t *>(std::get<1>(output_buffer2));
                    for (int j = 0; j < output_size2; j++)
                    {
                        /*FP16 to FP32 conversion*/
                        drpai_output_buf2[j + size_count2] = float16_to_float32(data_ptr2[j]);
                    }
                }
                else if (InOutDataType::FLOAT32 == std::get<0>(output_buffer2))
                {
                    /*Output Data = std::get<1>(output_buffer)*/
                    float *data_ptr2 = reinterpret_cast<float *>(std::get<1>(output_buffer2));
                    for (int j = 0; j < output_size2; j++)
                    {
                        drpai_output_buf2[j + size_count2] = data_ptr2[j];
                    }
                }
                else
                {
                    std::cerr << "[ERROR] Output data type : not floating point." << std::endl;
                    ret = -1;
                    break;
                }
                size_count2 += output_size2;
            }
            if (ret != 0)
            {
                std::cerr << "[ERROR] DRP Inference Not working !!! " << std::endl;
                return -1;
            }

            R_Post_Proc(drpai_output_buf2, 2);
            auto t5 = std::chrono::high_resolution_clock::now();
            
            auto r_post_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
            float total_time = float(inf_duration/1000.0) + float(r_post_proc_time/1000.0) + float(pre_proc_time/1000.0); 
            sum_inf_time+=inf_duration/1000.0;
            sum_pre_poc_time+=pre_proc_time/1000.0;
            sum_post_proc_time+=r_post_proc_time/1000.0;
            sum_total_time+=total_time;
        }
        /* camera3 */
        else if(inf_num==3)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            in_param.pre_in_shape_w = IMAGE_WIDTH;
            in_param.pre_in_shape_h = IMAGE_HEIGHT;

            in_param.pre_in_addr = (uintptr_t) drpai_buf3->phy_addr;

            ret = preruntime.Pre(&in_param, &output_ptr3, &out_size3);
            if (0 < ret)
            {
                fprintf(stderr, "[ERROR] Failed to run Pre-processing Runtime Pre2()\n");
                return 0;
            }
            auto t1 = std::chrono::high_resolution_clock::now();

            auto pre_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

            /*Set Pre-processing output to be inference input. */
            runtime.SetInput(0, (float*)output_ptr3);
            auto t2 = std::chrono::high_resolution_clock::now();
            runtime.Run(drpai_freq);
            auto t3 = std::chrono::high_resolution_clock::now();
            auto inf_duration = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
            auto t4 = std::chrono::high_resolution_clock::now();

            /* Get the number of output of the target model. */
            output_num3 = runtime.GetNumOutput();

            size_count3 = 0;
            /*GetOutput loop*/
            for (i = 0; i < output_num3; i++)
            {
                /* output_buffer below is tuple, which is { data type, address of output data, number of elements } */
                output_buffer3 = runtime.GetOutput(i);
                /*Output Data Size = std::get<2>(output_buffer). */
                output_size3 = std::get<2>(output_buffer3);

                /*Output Data Type = std::get<0>(output_buffer)*/
                if (InOutDataType::FLOAT16 == std::get<0>(output_buffer3))
                {
                    /*Output Data = std::get<1>(output_buffer)*/
                    uint16_t *data_ptr3 = reinterpret_cast<uint16_t *>(std::get<1>(output_buffer3));
                    for (int j = 0; j < output_size3; j++)
                    {
                        /*FP16 to FP32 conversion*/
                        drpai_output_buf3[j + size_count3] = float16_to_float32(data_ptr3[j]);
                    }
                }
                else if (InOutDataType::FLOAT32 == std::get<0>(output_buffer3))
                {
                    /*Output Data = std::get<1>(output_buffer)*/
                    float *data_ptr3 = reinterpret_cast<float *>(std::get<1>(output_buffer3));
                    for (int j = 0; j < output_size3; j++)
                    {
                        drpai_output_buf3[j + size_count3] = data_ptr3[j];
                    }
                }
                else
                {
                    std::cerr << "[ERROR] Output data type : not floating point." << std::endl;
                    ret = -1;
                    break;
                }
                size_count3 += output_size3;
            }
            if (ret != 0)
            {
                std::cerr << "[ERROR] DRP Inference Not working !!! " << std::endl;
                return -1;
            }

            R_Post_Proc(drpai_output_buf3, 3);
            auto t5 = std::chrono::high_resolution_clock::now();
            
            auto r_post_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
            float total_time = float(inf_duration/1000.0) + float(r_post_proc_time/1000.0) + float(pre_proc_time/1000.0); 
            sum_inf_time+=inf_duration/1000.0;
            sum_pre_poc_time+=pre_proc_time/1000.0;
            sum_post_proc_time+=r_post_proc_time/1000.0;
            sum_total_time+=total_time;
        }
    }
    mtx1.unlock();
    return 0;
}

/*****************************************
* Function Name : get_drpai_start_addr
* Description   : Function to get the start address of DRPAImem.
* Arguments     : drpai_fd: DRP-AI file descriptor
* Return value  : If non-zero, DRP-AI memory start address.
*                 0 is failure.
******************************************/
uint64_t get_drpai_start_addr(int drpai_fd)
{
    int ret = 0;
    drpai_data_t drpai_data;

    errno = 0;

    /* Get DRP-AI Memory Area Address via DRP-AI Driver */
    ret = ioctl(drpai_fd , DRPAI_GET_DRPAI_AREA, &drpai_data);
    if (-1 == ret)
    {
        std::cerr << "[ERROR] Failed to get DRP-AI Memory Area : errno=" << errno << std::endl;
        return 0;
    }

    return drpai_data.address;
}

/*****************************************
* Function Name : init_drpai
* Description   : Function to initialize DRP-AI.
* Arguments     : drpai_fd: DRP-AI file descriptor
* Return value  : If non-zero, DRP-AI memory start address.
*                 0 is failure.
******************************************/
uint64_t init_drpai(int drpai_fd)
{
    int ret = 0;
    uint64_t drpai_addr = 0;

    /*Get DRP-AI memory start address*/
    drpai_addr = get_drpai_start_addr(drpai_fd);
    
    if (drpai_addr == 0)
    {
        return 0;
    }

    return drpai_addr;
}

/*****************************************
 * Function Name : query_device_status
 * Description   : function to check USB device is connected.
 * Arguments     : device_type: for USB,  specify "usb".
 *                              for MIPI, specify "CRU".
 * Return value  : 1 if succeeded
 *                -1 otherwise
 ******************************************/
int query_device_status(std::string device_type)
{
    std::string response_string;
    /* Linux command to be executed */
    const char* command = "v4l2-ctl --list-devices";
    /* Open a pipe to the command and execute it */ 
    FILE* pipe = popen(command, "r");
    if (!pipe) 
    {
        std::cerr << "[ERROR] Unable to open the pipe." << std::endl;
        return -1;
    }
    /* Read the command output line by line */
    char buffer[128];
    size_t found;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) 
    {
        std::string response = std::string(buffer);
        found = response.find(device_type);
        if (found != std::string::npos)
        {
            fgets(buffer, sizeof(buffer), pipe);
            response_string = std::string(buffer);
            device_paths.push_back(response_string);    
        } 
    }
    pclose(pipe);
    /* return media port*/
    return 1;
}

/*****************************************
* Function Name : R_Kbhit_Thread
* Description   : Executes the Keyboard hit thread (checks if enter key is hit)
* Arguments     : threadid = thread identification
* Return value  : -
******************************************/
void *R_Kbhit_Thread(void *threadid)
{
    /*Semaphore Variable*/
    int32_t kh_sem_check = 0;
    /*Variable to store the getchar() value*/
    int32_t c = 0;
    /*Variable for checking return value*/
    int8_t ret = 0;

    printf("Key Hit Thread Starting\n");

    printf("************************************************\n");
    printf("* Press ENTER key to quit. *\n");
    printf("************************************************\n");

    /*Set Standard Input to Non Blocking*/
    errno = 0;
    ret = fcntl(0, F_SETFL, O_NONBLOCK);
    if (-1 == ret)
    {
        fprintf(stderr, "[ERROR] Failed to run fctnl(): errno=%d\n", errno);
        goto err;
    }

    while(1)
    {
        /*Gets the Termination request semaphore value. If different then 1 Termination was requested*/
        /*Checks if sem_getvalue is executed wihtout issue*/
        errno = 0;
        ret = sem_getvalue(&terminate_req_sem, &kh_sem_check);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
            goto err;
        }
        /*Checks the semaphore value*/
        if (1 != kh_sem_check)
        {
            goto key_hit_end;
        }

        c = getchar();
        if (EOF != c)
        {
            /* When key is pressed. */
            printf("key Detected.\n");
            goto err;
        }
        else
        {
            /* When nothing is pressed. */
            usleep(WAIT_TIME);
        }
    }

/*Error Processing*/
err:
    /*Set Termination Request Semaphore to 0*/
    sem_trywait(&terminate_req_sem);
    goto key_hit_end;

key_hit_end:
    printf("Key Hit Thread Terminated\n");
    pthread_exit(NULL);
}

/*****************************************
* Function Name : R_Capture_Thread
* Description   : Executes the V4L2 capture with Capture thread.
* Arguments     : threadid = thread identification
* Return value  : -
******************************************/
void *R_Capture_Thread(void *threadid)
{
    /*Semaphore Variable*/
    int32_t capture_sem_check = 0;
    int8_t ret,i = 0;

    printf("Capture Thread Starting\n");

    cap.resize(number_of_cameras);
    cap_frame.resize(number_of_cameras);
    g_frame.resize(number_of_cameras);

    for(i=0;i<number_of_cameras;i++)
    {
        cap[i].open(gstreamer_pipeline[i], CAP_GSTREAMER);
        if (!cap[i].isOpened())
        {
        std::cerr << "[ERROR] Error opening video stream  or camera " << device_paths[i] << std::endl;
        return 0;
        } 
    }

    while(1)
    {
        while(1)
        {
            /*Gets the Termination request semaphore value. If different then 1 Termination was requested*/
            /*Checks if sem_getvalue is executed wihtout issue*/
            errno = 0;
            ret = sem_getvalue(&terminate_req_sem, &capture_sem_check);
            if (0 != ret)
            {
                fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
                goto err;
            }
            /*Checks the semaphore value*/
            if (1 != capture_sem_check)
            {
                goto capture_end;
            }
            for(i=0;i<number_of_cameras;i++)
            {
            cap[i].read(cap_frame[i]); 
            if (cap_frame[i].empty())
            {
                std::cout << "[INFO] Video ended or corrupted frame from "<< device_paths[i] <<endl;
                return 0;
            }
            }
            /*Checks if image frame from Capture Thread is ready.*/
            if (capture_start.load())
            {
                break;
            }
            usleep(WAIT_TIME);
        }
        for(i=0;i<number_of_cameras;i++)
        {
            g_frame[i]=cap_frame[i].clone();
        }
        if(flip_mode == 1)
        {
            for(i=0;i<number_of_cameras;i++)
            {
                cv::flip(g_frame[i],g_frame[i],  1);
            }
        }
        capture_start.store(0);
    } /*End of Loop*/

/*Error Processing*/
err:
    sem_trywait(&terminate_req_sem);
    goto capture_end;

capture_end:
    printf("Capture Thread Terminated\n");
    pthread_exit(NULL);
}

void *R_Memcpy_Thread(void *threadid)
{
    int32_t ret,i = 0;
    int32_t memcpy_sem_check = 0;

    static int8_t memcpy_flag = 1;
    
    std::vector<Mat> input_images;
    input_images.resize(number_of_cameras);
    
    printf("Memory copy Loop Starting\n");
    /*Memory copy Loop Start*/
    while(1)
    {
            /*Gets the Termination request semaphore value. If different then 1 Termination was requested*/
            /*Checks if sem_getvalue is executed wihtout issue*/
            errno = 0;
            ret = sem_getvalue(&terminate_req_sem, &memcpy_sem_check);
            if (0 != ret)
            {
                fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
                goto err;
            }
            /*Checks the semaphore value*/
            if (1 != memcpy_sem_check)
            {
                goto memcpy_end;
            }

            if (!capture_start.load() && !img_processing_start.load() && !inference_start.load() && memcpy_flag == 1)
            {
                /* Copy captured image to inference buffer. This will be used in AI Inference Thread. */

                inf_frames.resize(number_of_cameras);
                for(i=0;i<number_of_cameras;i++)
                {
                   memcpy(inf_frames[i].data, g_frame[i].data,  IMAGE_WIDTH  * IMAGE_HEIGHT  * BGR_CHANNEL);
                   input_images[i] = inf_frames[i].clone();
                }
                if(number_of_cameras == 4)
                {
                /*Copy input data to drpai_buf for DRP-AI Pre-processing Runtime.*/
                memcpy( drpai_buf->mem,  input_images[0].data, drpai_buf->size);
                memcpy( drpai_buf1->mem, input_images[1].data, drpai_buf1->size);
                memcpy( drpai_buf2->mem, input_images[2].data, drpai_buf2->size);
                memcpy( drpai_buf3->mem, input_images[3].data, drpai_buf3->size);
                }
                else if(number_of_cameras == 3)
                {
                memcpy( drpai_buf->mem,  input_images[0].data,  drpai_buf->size);
                memcpy( drpai_buf1->mem, input_images[1].data, drpai_buf1->size);
                memcpy( drpai_buf2->mem, input_images[2].data, drpai_buf2->size);
                }
                else if(number_of_cameras == 2)
                {
                memcpy( drpai_buf->mem,  input_images[0].data,  drpai_buf->size);
                memcpy( drpai_buf1->mem, input_images[1].data, drpai_buf1->size);   
                }
                else
                {
                memcpy( drpai_buf->mem,  input_images[0].data,  drpai_buf->size);
                }

                /* Flush buffer */
                ret = buffer_flush_dmabuf(drpai_buf->idx, drpai_buf->size);
                if (0 != ret)
                {
                    goto err;
                }
                ret = buffer_flush_dmabuf(drpai_buf1->idx, drpai_buf1->size);
                if (0 != ret)
                {
                    goto err;
                }
                ret = buffer_flush_dmabuf(drpai_buf2->idx, drpai_buf2->size);
                if (0 != ret)
                {
                    goto err;
                }
                ret = buffer_flush_dmabuf(drpai_buf3->idx, drpai_buf3->size);
                if (0 != ret)
                {
                    goto err;
                }

                inference_start.store(1); /* Flag for AI Inference Thread. */
                img_processing_start.store(1);

                memcpy_flag = 0;
            }

            if (!img_processing_start.load() && !inference_start.load() && memcpy_flag == 0)
            {
                /* Copy captured image to inference buffer. This will be used in AI Inference Thread. */
                for(i=0;i<number_of_cameras;i++)
                {
                memcpy(img_frames[i].data,  inf_frames[i].data,  IMAGE_WIDTH  * IMAGE_HEIGHT  * BGR_CHANNEL);
                }
                capture_start.store(1); /* Flag for AI Inference Thread. */

                memcpy_flag = 1;
            }
        /*Wait for 1 TICK.*/
        usleep(WAIT_TIME);
    }/*End of Inference Loop*/

/*Error Processing*/
err:
    /*Set Termination Request Semaphore to 0*/
    sem_trywait(&terminate_req_sem);
    goto memcpy_end;
/*Memory Copy Thread Termination*/
memcpy_end:
    /*To terminate the loop in AI Inference Thread.*/
    inference_start.store(0);

    printf("Memory Copy Thread Terminated\n");
    pthread_exit(NULL);
}

void *R_Inf_Thread(void *threadid)
{
    int32_t ret = 0;
    int32_t inf_sem_check = 0;

    printf("Inference Loop Starting\n");
    /*Inference Loop Start*/
    while(1)
    {
        while(1)
        {
            /*Gets the Termination request semaphore value. If different then 1 Termination was requested*/
            /*Checks if sem_getvalue is executed wihtout issue*/
            errno = 0;
            ret = sem_getvalue(&terminate_req_sem, &inf_sem_check);
            if (0 != ret)
            {
                fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
                goto err;
            }
            /*Checks the semaphore value*/
            if (1 != inf_sem_check)
            {
                goto ai_inf_end;
            }
            /*Checks if image frame from Capture Thread is ready.*/
            if (inference_start.load())
            {
                break;
            }
            usleep(WAIT_TIME);
        }

        int ret = Vehicle_Detection();
        if (ret != 0)
        {
            std::cerr << "[ERROR] Inference Not working !!! " << std::endl;
            goto err;
        }
        inference_start.store(0);
    }/*End of Inference Loop*/

/*Error Processing*/
err:
    /*Set Termination Request Semaphore to 0*/
    sem_trywait(&terminate_req_sem);
    goto ai_inf_end;
/*AI Thread Termination*/
ai_inf_end:
    /*To terminate the loop in Capture Thread.*/
    printf("AI Inference Thread Terminated\n");
    pthread_exit(NULL);
}


int8_t R_Main_Process()
{
    /*Main Process Variables*/
    int8_t main_ret,i = 0;
    /*Semaphore Related*/
    int32_t main_sem_check = 0;
    /*Variable for checking return value*/
    int8_t ret = 0;

    stringstream stream;
    string str = "";
    int32_t baseline = 10;
    cv::Mat output_image = cv::Mat(DISP_OUTPUT_HEIGHT,DISP_OUTPUT_WIDTH , CV_8UC3);
    cv::Mat bgra_image = cv::Mat(DISP_OUTPUT_HEIGHT,DISP_OUTPUT_WIDTH,CV_8UC4);
    uint8_t * img_buffer0;

    string sensor_str = "";

    img_buffer0 = (unsigned char*) (malloc(DISP_OUTPUT_WIDTH*DISP_OUTPUT_HEIGHT*BGRA_CHANNEL));

    printf("Main Loop Starts\n");
    /*Main Loop Start*/
    while(1)
    {
        while(1)
        {
            /*Gets the Termination request semaphore value. If different then 1 Termination was requested*/
            /*Checks if sem_getvalue is executed wihtout issue*/
            errno = 0;
            ret = sem_getvalue(&terminate_req_sem, &main_sem_check);
            if (0 != ret)
            {
                fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
                goto err;
            }
            /*Checks the semaphore value*/
            if (1 != main_sem_check)
            {
                goto main_proc_end;
            }
            /*Checks if image frame from Capture Thread is ready.*/
            if (img_processing_start.load())
            {
                break;
            }
            usleep(WAIT_TIME);
        }

        output_image.setTo(cv::Scalar(0, 0, 0));

        /* Draw bounding box and Mosaic on the frame */
        draw_bounding_box();

        /* Display TSU value */
        FILE *fp;
        char buff[16]="";
        float tmp;
        fp = fopen("/sys/class/thermal/thermal_zone1/temp", "r");
        fgets(buff, 16, fp);
        fclose(fp);
        tmp  = (float)atoi(buff)/1000;

        if(flip_mode==1)
        {
            if(number_of_cameras == 4)
            {
                img_frames[0].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH, FIRST_FRAME_Y_COORDINATE, IMAGE_WIDTH, IMAGE_HEIGHT)));
                img_frames[1].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE, FIRST_FRAME_Y_COORDINATE, IMAGE_WIDTH, IMAGE_HEIGHT)));
                img_frames[2].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT)));
                img_frames[3].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT)));
                stream.str("");
                stream << "Cam 1";
                str = stream.str();
                Size cam_name_size = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE - cam_name_size.width -15 ,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE,FIRST_FRAME_Y_COORDINATE+cam_name_size.height +15), cv::Scalar(0, 0, 255), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE - cam_name_size.width -5), (FIRST_FRAME_Y_COORDINATE +cam_name_size.height +5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(255, 255, 255), CAM_NAME_THICKNESS);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE, FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH-3,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT-3), cv::Scalar(0, 0, 255), 2);
                stream.str("");
                stream << "Cam 0";
                str = stream.str();
                Size cam_name_size1 = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH ,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH+ cam_name_size1.width +15 ,FIRST_FRAME_Y_COORDINATE+cam_name_size1.height +15), cv::Scalar(0, 255, 255), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH +5), (FIRST_FRAME_Y_COORDINATE +cam_name_size1.height +5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(0, 0, 0), CAM_NAME_THICKNESS);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH, FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT-3), cv::Scalar(0, 255, 255), 2);
                stream.str("");
                stream << "Cam 3";
                str = stream.str();
                Size cam_name_size2 = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE - cam_name_size2.width -15 ,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT), cv::Point(FIRST_FRAME_X_COORDINATE,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+cam_name_size2.height +15), cv::Scalar(0, 128, 255), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE - cam_name_size2.width -5), (FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+ 5 + cam_name_size2.height)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(0, 0, 0), CAM_NAME_THICKNESS);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE, FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+1), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH-3,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+IMAGE_HEIGHT+1), cv::Scalar(0, 128, 255), 2);
                stream.str("");
                stream << "Cam 2";
                str = stream.str();
                Size cam_name_size3 = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH+ cam_name_size3.width +15 ,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+cam_name_size3.height +15), cv::Scalar(150, 0, 0), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH+5), (FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+ cam_name_size3.height+5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(255, 255, 255), CAM_NAME_THICKNESS);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+IMAGE_HEIGHT), cv::Scalar(150, 0, 0), 2);
            } 
            else if(number_of_cameras == 3)
            {
                img_frames[0].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH, FIRST_FRAME_Y_COORDINATE, IMAGE_WIDTH, IMAGE_HEIGHT)));
                img_frames[1].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE, FIRST_FRAME_Y_COORDINATE, IMAGE_WIDTH, IMAGE_HEIGHT)));
                img_frames[2].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT)));
                stream.str("");
                stream << "Cam 1";
                str = stream.str();
                Size cam_name_size = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE - cam_name_size.width -15 ,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE,FIRST_FRAME_Y_COORDINATE+cam_name_size.height +15), cv::Scalar(0, 0, 255), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE - cam_name_size.width -5), (FIRST_FRAME_Y_COORDINATE +cam_name_size.height +5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(255, 255, 255), CAM_NAME_THICKNESS);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE, FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH-3,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT-3), cv::Scalar(0, 0, 255), 2);
                stream.str("");
                stream << "Cam 0";
                str = stream.str();
                Size cam_name_size1 = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH ,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH+ cam_name_size1.width +15 ,FIRST_FRAME_Y_COORDINATE+cam_name_size1.height +15), cv::Scalar(0, 255, 255), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH +5), (FIRST_FRAME_Y_COORDINATE +cam_name_size1.height +5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(0, 0, 0), CAM_NAME_THICKNESS);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH, FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT-3), cv::Scalar(0, 255, 255), 2);
                stream.str("");
                stream << "Cam 2";
                str = stream.str();
                Size cam_name_size3 = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH+ cam_name_size3.width +15 ,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+cam_name_size3.height +15), cv::Scalar(150, 0, 0), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH+5), (FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+ cam_name_size3.height+5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(255, 255, 255), CAM_NAME_THICKNESS);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+IMAGE_HEIGHT), cv::Scalar(150, 0, 0), 2);                                               
            }
            else if(number_of_cameras == 2)
            {
                img_frames[0].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH, FIRST_FRAME_Y_COORDINATE, IMAGE_WIDTH, IMAGE_HEIGHT)));
                img_frames[1].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE, FIRST_FRAME_Y_COORDINATE, IMAGE_WIDTH, IMAGE_HEIGHT)));
                stream.str("");
                stream << "Cam 1";
                str = stream.str();
                Size cam_name_size = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE - cam_name_size.width -15 ,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE,FIRST_FRAME_Y_COORDINATE+cam_name_size.height +15), cv::Scalar(0, 0, 255), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE - cam_name_size.width -5), (FIRST_FRAME_Y_COORDINATE +cam_name_size.height +5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(255, 255, 255), CAM_NAME_THICKNESS);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE, FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH-3,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT-3), cv::Scalar(0, 0, 255), 2);
                stream.str("");
                stream << "Cam 0";
                str = stream.str();
                Size cam_name_size1 = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH ,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH+ cam_name_size1.width +15 ,FIRST_FRAME_Y_COORDINATE+cam_name_size1.height +15), cv::Scalar(0, 255, 255), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH +5), (FIRST_FRAME_Y_COORDINATE +cam_name_size1.height +5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(0, 0, 0), CAM_NAME_THICKNESS);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH, FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT-3), cv::Scalar(0, 255, 255), 2);
            }
            else
            {
                img_frames[0].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH, FIRST_FRAME_Y_COORDINATE, IMAGE_WIDTH, IMAGE_HEIGHT)));
                stream.str("");
                stream << "Cam 0";
                str = stream.str();
                Size cam_name_size1 = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH ,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH+ cam_name_size1.width +15 ,FIRST_FRAME_Y_COORDINATE+cam_name_size1.height +15), cv::Scalar(0, 255, 255), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH +5), (FIRST_FRAME_Y_COORDINATE +cam_name_size1.height +5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(0, 0, 0), CAM_NAME_THICKNESS);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH, FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT-3), cv::Scalar(0, 255, 255), 2);
            }
        }
        else
        {
            if(number_of_cameras == 4)
            {
                img_frames[0].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE, FIRST_FRAME_Y_COORDINATE, IMAGE_WIDTH, IMAGE_HEIGHT)));
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH-3,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT-3), cv::Scalar(0, 255, 255), FRAME_THICKNESS);

                img_frames[1].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE , IMAGE_WIDTH, IMAGE_HEIGHT)));
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT-3), cv::Scalar(0, 0, 255), FRAME_THICKNESS);

                img_frames[2].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT)));
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE, FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+1), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH-3,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+IMAGE_HEIGHT+1), cv::Scalar(150, 0, 0), FRAME_THICKNESS);

                img_frames[3].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT ,IMAGE_WIDTH, IMAGE_HEIGHT)));
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+IMAGE_HEIGHT), cv::Scalar(0, 128, 255), FRAME_THICKNESS);

                stream.str("");
                stream << "Cam 0";
                str = stream.str();
                Size cam_name_size = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE - cam_name_size.width -15 ,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE,FIRST_FRAME_Y_COORDINATE+cam_name_size.height +15), cv::Scalar(0, 255, 255), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE - cam_name_size.width -5), (FIRST_FRAME_Y_COORDINATE +cam_name_size.height +5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(0, 0, 0), CAM_NAME_THICKNESS);
                
                stream.str("");
                stream << "Cam 1";
                str = stream.str();
                Size cam_name_size1 = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH ,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH+ cam_name_size1.width +15 ,FIRST_FRAME_Y_COORDINATE+cam_name_size1.height +15), cv::Scalar(0, 0, 255), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH +5), (FIRST_FRAME_Y_COORDINATE+cam_name_size1.height +5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(255, 255, 255),CAM_NAME_THICKNESS);
                
                stream.str("");
                stream << "Cam 2";
                str = stream.str();
                Size cam_name_size2 = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE - cam_name_size2.width -15 ,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT), cv::Point(FIRST_FRAME_X_COORDINATE,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+cam_name_size2.height +15), cv::Scalar(150, 0, 0), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE - cam_name_size2.width -5), (FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+cam_name_size2.height+5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(255, 255, 255), CAM_NAME_THICKNESS);
                
                stream.str("");
                stream << "Cam 3";
                str = stream.str();
                Size cam_name_size3 = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH+ cam_name_size3.width +15 ,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+cam_name_size3.height +15), cv::Scalar(0, 128, 255), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH+5), (FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+ cam_name_size3.height+5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(0, 0, 0), CAM_NAME_THICKNESS);
                
            } 
            else if(number_of_cameras == 3)
            {
               img_frames[0].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE, FIRST_FRAME_Y_COORDINATE, IMAGE_WIDTH, IMAGE_HEIGHT)));
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH-3,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT-3), cv::Scalar(0, 255, 255), FRAME_THICKNESS);

                img_frames[1].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE , IMAGE_WIDTH, IMAGE_HEIGHT)));
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT-3), cv::Scalar(0, 0, 255), FRAME_THICKNESS);

                img_frames[2].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT)));
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE, FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+1), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH-3,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+IMAGE_HEIGHT+1), cv::Scalar(150, 0, 0), FRAME_THICKNESS);

                stream.str("");
                stream << "Cam 0";
                str = stream.str();
                Size cam_name_size = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE - cam_name_size.width -15 ,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE,FIRST_FRAME_Y_COORDINATE+cam_name_size.height +15), cv::Scalar(0, 255, 255), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE - cam_name_size.width -5), (FIRST_FRAME_Y_COORDINATE +cam_name_size.height +5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(0, 0, 0), CAM_NAME_THICKNESS);
                
                stream.str("");
                stream << "Cam 1";
                str = stream.str();
                Size cam_name_size1 = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH ,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH+ cam_name_size1.width +15 ,FIRST_FRAME_Y_COORDINATE+cam_name_size1.height +15), cv::Scalar(0, 0, 255), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH +5), (FIRST_FRAME_Y_COORDINATE+cam_name_size1.height +5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(255, 255, 255),CAM_NAME_THICKNESS);
                
                stream.str("");
                stream << "Cam 2";
                str = stream.str();
                Size cam_name_size2 = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE - cam_name_size2.width -15 ,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT), cv::Point(FIRST_FRAME_X_COORDINATE,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+cam_name_size2.height +15), cv::Scalar(150, 0, 0), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE - cam_name_size2.width -5), (FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+cam_name_size2.height+5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(255, 255, 255), CAM_NAME_THICKNESS);
                                                      
            }
            else if(number_of_cameras == 2)
            {
                img_frames[0].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE, FIRST_FRAME_Y_COORDINATE, IMAGE_WIDTH, IMAGE_HEIGHT)));
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH-3,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT-3), cv::Scalar(0, 255, 255), FRAME_THICKNESS);

                img_frames[1].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE , IMAGE_WIDTH, IMAGE_HEIGHT)));
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT-3), cv::Scalar(0, 0, 255), FRAME_THICKNESS);

                stream.str("");
                stream << "Cam 0";
                str = stream.str();
                Size cam_name_size = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE - cam_name_size.width -15 ,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE,FIRST_FRAME_Y_COORDINATE+cam_name_size.height +15), cv::Scalar(0, 255, 255), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE - cam_name_size.width -5), (FIRST_FRAME_Y_COORDINATE +cam_name_size.height +5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(0, 0, 0), CAM_NAME_THICKNESS);
                
                stream.str("");
                stream << "Cam 1";
                str = stream.str();
                Size cam_name_size1 = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH ,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH+ cam_name_size1.width +15 ,FIRST_FRAME_Y_COORDINATE+cam_name_size1.height +15), cv::Scalar(0, 0, 255), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH+IMAGE_WIDTH +5), (FIRST_FRAME_Y_COORDINATE+cam_name_size1.height +5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(255, 255, 255),CAM_NAME_THICKNESS);
                                                                  
            }
            else
            {
                img_frames[0].copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE, FIRST_FRAME_Y_COORDINATE, IMAGE_WIDTH, IMAGE_HEIGHT)));
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE+IMAGE_WIDTH-3,FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT-3), cv::Scalar(0, 255, 255), FRAME_THICKNESS);

                stream.str("");
                stream << "Cam 0";
                str = stream.str();
                Size cam_name_size = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, CAM_NAME_THICKNESS, &baseline);
                cv::rectangle(output_image, cv::Point(FIRST_FRAME_X_COORDINATE - cam_name_size.width -15 ,FIRST_FRAME_Y_COORDINATE), cv::Point(FIRST_FRAME_X_COORDINATE,FIRST_FRAME_Y_COORDINATE+cam_name_size.height +15), cv::Scalar(0, 255, 255), cv::FILLED);
                putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE - cam_name_size.width -5), (FIRST_FRAME_Y_COORDINATE +cam_name_size.height +5)), FONT_HERSHEY_TRIPLEX, 
                                CHAR_SCALE_APP_NAME, Scalar(0, 0, 0), CAM_NAME_THICKNESS);
                                        
            }
        }
        stream.str("");
        stream << "Multi camera vehicle detection";
        str = stream.str();
        Size app_name_size = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_APP_NAME, APP_NAME_THICKNESS, &baseline);
        putText(output_image, str,Point((APP_NAME_X), (APP_NAME_Y+ app_name_size.height)), FONT_HERSHEY_TRIPLEX, 
                        CHAR_SCALE_APP_NAME, Scalar(255, 255, 255), APP_NAME_THICKNESS);
        mtx1.lock();
        stream.str("");
        stream << "Total time: " << fixed << setprecision(1) << sum_total_time <<" ms";
        str = stream.str();
        Size tot_time_size = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_LARGE, TIME_THICKNESS, &baseline);
        putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE - tot_time_size.width -LEFT_ALIGN_OFFSET ), (FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+ HALF_IMAGE_HEIGHT+ tot_time_size.height)), FONT_HERSHEY_TRIPLEX, 
                        CHAR_SCALE_LARGE, Scalar(0, 128, 255), TIME_THICKNESS);
                        
        stream.str("");
        stream << "Pre-proc: " << fixed << setprecision(1)<< sum_pre_poc_time<<" ms";
        str = stream.str();
        Size pre_proc_size = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_SMALL, TIME_THICKNESS, &baseline);
        putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE - pre_proc_size.width -LEFT_ALIGN_OFFSET ), (FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+ HALF_IMAGE_HEIGHT+ tot_time_size.height + 40+ pre_proc_size.height)), FONT_HERSHEY_TRIPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), TIME_THICKNESS);

        stream.str("");
        stream << "Inference: "<< fixed << setprecision(1) << sum_inf_time<<" ms";
        str = stream.str();
        Size inf_size = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_SMALL, TIME_THICKNESS, &baseline);
        putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE - inf_size.width -LEFT_ALIGN_OFFSET), (FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+ HALF_IMAGE_HEIGHT+ tot_time_size.height + 40 + pre_proc_size.height + 10 + inf_size.height)), FONT_HERSHEY_TRIPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), TIME_THICKNESS);
        
        stream.str("");
        stream << "Post-proc: "<< fixed << setprecision(1) << sum_post_proc_time<<" ms";
        str = stream.str();
        Size post_proc_size = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_SMALL, TIME_THICKNESS, &baseline);
        putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE - post_proc_size.width - LEFT_ALIGN_OFFSET), (FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+ HALF_IMAGE_HEIGHT+ tot_time_size.height + 40 + pre_proc_size.height + 10 + inf_size.height+10+post_proc_size.height)), FONT_HERSHEY_TRIPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), TIME_THICKNESS);
        mtx1.unlock();
        stream.str("");
        stream << "Temperature: "<< fixed <<setprecision(1) << tmp << "C";
        str = stream.str();
        Size temp_size = getTextSize(str, FONT_HERSHEY_TRIPLEX,CHAR_SCALE_LARGE, TIME_THICKNESS, &baseline);
        putText(output_image, str,Point((FIRST_FRAME_X_COORDINATE - temp_size.width - LEFT_ALIGN_OFFSET), ( FIRST_FRAME_Y_COORDINATE+IMAGE_HEIGHT+ HALF_IMAGE_HEIGHT+ tot_time_size.height + 40 + pre_proc_size.height + 10 + inf_size.height+10+post_proc_size.height+40 + temp_size.height)), FONT_HERSHEY_TRIPLEX,
                    CHAR_SCALE_LARGE, Scalar(255, 255, 255), TIME_THICKNESS);
        cv::cvtColor(output_image, bgra_image, cv::COLOR_BGR2BGRA);

        memcpy(img_buffer0, bgra_image.data, DISP_OUTPUT_WIDTH * DISP_OUTPUT_HEIGHT * BGRA_CHANNEL);
        wayland.commit(img_buffer0, NULL);

        img_processing_start.store(0);
    }/*End of Loop*/
    goto main_proc_end;
/*Error Processing*/
err:
    sem_trywait(&terminate_req_sem);
    main_ret = 1;
    goto main_proc_end;
/*Main Processing Termination*/
main_proc_end:
    printf("Main Process Terminated\n");
    free(img_buffer0);
    return main_ret;
}

int main(int argc, char *argv[])
{
    /*Multithreading Variables*/
    int32_t create_thread_ai  = -1;
    int32_t create_thread_key = -1;
    int32_t create_thread_capture = -1;
    int32_t create_thread_memcpy  = -1;
    int8_t ret_main = 0;
    int32_t ret = 0;
    int8_t main_proc = 0;
    int32_t sem_create = -1;
    std::string command,white_space;
    int32_t i,status = 0;
    int x=6;
    int y=7;
    std::string input_source = argv[1];
    std::cout << "Starting Multi camera vehicle detection application" << std::endl;

    InOutDataType input_data_type;
    bool runtime_status = false;
    int drpai_fd;

    errno = 0;
    drpai_fd = open("/dev/drpai0", O_RDWR);
    if (0 > drpai_fd)
    {
        std::cerr << "[ERROR] Failed to open DRP-AI Driver : errno=" << errno << std::endl;
        goto end_close_drpai;
    }

    unsigned long OCA_list[16];
    /*Disable OpenCV Accelerator due to the use of multithreading */
    for (int i=0; i<16; i++)
    {
        OCA_list[i] = 0;    //disable
    }
    OCA_Activate( &OCA_list[0] );

    if (strcmp(argv[1],"USB")==0 || strcmp(argv[1],"MIPI")==0)
    {   
        if(argc ==2)
        {
            cout<<"enter number of cameras as third argument";
            goto end_close_drpai;
        }
        else if(argc == 3)
        {                      
            if(strcmp(argv[1],"USB")==0)
            {
                if(strcmp(argv[2],"1")==0 || strcmp(argv[2],"2")==0 || strcmp(argv[2],"3")==0)
                {
                    number_of_cameras=atoi(argv[2]);
                    flip_mode=0;
                    drpai_freq = DRPAI_FREQ;
                }
                else
                {
                    cout<<"enter number of cameras as third argument maximum number to enter is 3 and minimum is 1"<<endl;
                    goto end_close_drpai;
                }
            }
            else
            {
                if(strcmp(argv[2],"1")==0 || strcmp(argv[2],"2")==0 || strcmp(argv[2],"3")==0 || strcmp(argv[2],"4")==0)
                {
                    number_of_cameras=atoi(argv[2]);
                    flip_mode=0;
                    drpai_freq = DRPAI_FREQ;
                }
                else
                {
                    cout<<"enter number of cameras as third argument maximum number to enter is 4 and minimum is 1"<<endl;
                    goto end_close_drpai;
                }
            }
        }
        else if (argc == 4)
        {
            if(strcmp(argv[3],"FLIP")==0)
            {
                flip_mode=1;  
                if(strcmp(argv[1],"USB")==0)  
                {
                    if(strcmp(argv[2],"1")==0 || strcmp(argv[2],"2")==0 || strcmp(argv[2],"3")==0 )
                    {
                    number_of_cameras=atoi(argv[2]);
                    drpai_freq = DRPAI_FREQ;
                    }
                    else
                    {
                    cout<<"enter number of cameras as third argument and maximum number to enter is 3 and minimum is 1"<<endl;
                    goto end_close_drpai;
                    }
                }   
                else
                {
                    if(strcmp(argv[2],"1")==0 || strcmp(argv[2],"2")==0 || strcmp(argv[2],"3")==0 || strcmp(argv[2],"4")==0 )
                    {
                    number_of_cameras=atoi(argv[2]);
                    drpai_freq = DRPAI_FREQ;
                    }
                    else
                    {
                    cout<<"enter number of cameras as third argument and maximum number to enter is 4 and minimum is 1"<<endl;
                    goto end_close_drpai;
                    }
                }         
            }
            else
            {
                if((atoi(argv[3])<= (pow(2,sizeof(long long int) * 8 -1) -1) ) && ( atoi(argv[3]) >= -(pow(2,sizeof(long long int) * 8 -1))))
                {
                    drpai_freq = atoi(argv[3]);
                    if ((1 <= drpai_freq) && (127 >= drpai_freq))
                    {
                        flip_mode=0;
                        if(strcmp(argv[1],"USB")==0)
                        {
                            if(strcmp(argv[2],"1")==0 || strcmp(argv[2],"2")==0 || strcmp(argv[2],"3")==0)
                            {
                            number_of_cameras=atoi(argv[2]);
                            }
                            else
                            {
                            cout<<"enter number of cameras as third argument and maximum number to enter is 3 and minimum is 1"<<endl;
                            goto end_close_drpai;
                            }
                        }
                        else
                        {
                            if(strcmp(argv[2],"1")==0 || strcmp(argv[2],"2")==0 || strcmp(argv[2],"3")==0 || strcmp(argv[2],"4")==0)
                            {
                            number_of_cameras=atoi(argv[2]);
                            }
                            else
                            {
                            cout<<"enter number of cameras as third argument and maximum number to enter is 4 and minimum is 1"<<endl;
                            goto end_close_drpai;
                            }
                        }
                    }
                    else
                    {
                        cout<<"enter fourth argument as FLIP if you want or enter frequency [1,127]"<<endl; 
                        goto end_close_drpai;
                    }

                }
                else
                {
                    cout<<"enter integer in range"<<endl;
                    goto end_close_drpai;
                }
            }
        }
        else if (argc == 5)
        {
            if(strcmp(argv[1],"USB")==0)
            {
                if(strcmp(argv[2],"1")==0 || strcmp(argv[2],"2")==0 || strcmp(argv[2],"3")==0)
                {
                    number_of_cameras=atoi(argv[2]);
                }
                else
                {
                    cout<<"enter number of cameras as third argument and maximum number to enter is 3 and minimum is 1"<<endl;
                    goto end_close_drpai;
                }
            }
            else
            {
                if(strcmp(argv[2],"1")==0 || strcmp(argv[2],"2")==0 || strcmp(argv[2],"3")==0 || strcmp(argv[2],"4")==0)
                {
                    number_of_cameras=atoi(argv[2]);
                }
                else
                {
                    cout<<"enter number of cameras as third argument and maximum number to enter is 4 and minimum is 1"<<endl;
                    goto end_close_drpai;
                }
            }
            if(strcmp(argv[3],"FLIP")==0)
            {
                flip_mode=1;
            }
            else
            {
                cout<<"support for FLIP"<<endl;
                goto end_close_drpai;
            }

            if((atoi(argv[4])<= (pow(2,sizeof(long long int) * 8 -1) -1) ) && ( atoi(argv[4]) >= -(pow(2,sizeof(long long int) * 8 -1))))
            {
                if ((1 <= atoi(argv[4])) && (127 >= atoi(argv[4])))
                {
                    drpai_freq = atoi(argv[4]);
                }
                else
                {
                    cout<<"enter fifth argument to be drp-ai frequency between [1,127] "<<endl;
                    goto end_close_drpai;
                }

            }
            else 
            {
                cout<<"enter integer in range"<<endl;
                goto end_close_drpai;
            } 
        }
        else
        {
            cout<<"wrong number of parameters passed"<<endl;
            goto end_close_drpai;
        }

    }
    else
    {
        std::cout<<"Support for USB mode or MIPI mode only."<<std::endl;
        goto end_close_drpai;
    }

    /*Initialzie DRP-AI (Get DRP-AI memory address and set DRP-AI frequency)*/
    drpaimem_addr_start = init_drpai(drpai_fd);
    if (drpaimem_addr_start == 0)
    {
        close(drpai_fd);
        goto end_close_drpai;
    }
    /*Load pre_dir object to DRP-AI */
    ret = preruntime.Load(pre_dir);
    if (0 < ret)
    {
        fprintf(stderr, "[ERROR] Failed to run Pre-processing Runtime Load().\n");
        ret_main = -1;
        goto end_close_drpai;
    }

    /*Load model_dir structure and its weight to runtime object */
    runtime_status = runtime.LoadModel(model_dir, drpaimem_addr_start + DRPAI_MEM_OFFSET);
    if(!runtime_status)
    {
        std::cerr << "[ERROR] Failed to load model. " << std::endl;
        close(drpai_fd);
        goto end_close_drpai;
    }

    /*Get input data */
    input_data_type = runtime.GetInputDataType(0);
    if (InOutDataType::FLOAT32 == input_data_type)
    {
        /*Do nothing*/
    }
    else if (InOutDataType::FLOAT16 == input_data_type)
    {
        fprintf(stderr, "[ERROR] Input data type : FP16.\n");
        /*If your model input data type is FP16, use std::vector<uint16_t> for reading input data. */
        goto end_close_drpai;
    }
    else
    {
        fprintf(stderr, "[ERROR] Input data type : neither FP32 nor FP16.\n");
        goto end_close_drpai;
    }
    /*Initialize buffer for DRP-AI Pre-processing Runtime. */
    drpai_buf = (dma_buffer*)malloc(sizeof(dma_buffer));
    ret = buffer_alloc_dmabuf(drpai_buf,  IMAGE_WIDTH  * IMAGE_HEIGHT * BGR_CHANNEL);
    if (-1 == ret)
    {
        fprintf(stderr, "[ERROR] Failed to Allocate DMA buffer for the drpai_buf\n");
        goto end_free_malloc;
    }

    drpai_buf1 = (dma_buffer*)malloc(sizeof(dma_buffer));
    ret = buffer_alloc_dmabuf(drpai_buf1, IMAGE_WIDTH  * IMAGE_HEIGHT * BGR_CHANNEL);
    if (-1 == ret)
    {
        fprintf(stderr, "[ERROR] Failed to Allocate DMA buffer for the drpai_buf\n");
        goto end_free_malloc;
    }

    drpai_buf2 = (dma_buffer*)malloc(sizeof(dma_buffer));
    ret = buffer_alloc_dmabuf(drpai_buf2, IMAGE_WIDTH  * IMAGE_HEIGHT * BGR_CHANNEL);
    if (-1 == ret)
    {
        fprintf(stderr, "[ERROR] Failed to Allocate DMA buffer for the drpai_buf\n");
        goto end_free_malloc;
    }

    drpai_buf3 = (dma_buffer*)malloc(sizeof(dma_buffer));
    ret = buffer_alloc_dmabuf(drpai_buf3, IMAGE_WIDTH  * IMAGE_HEIGHT * BGR_CHANNEL);
    if (-1 == ret)
    {
        fprintf(stderr, "[ERROR] Failed to Allocate DMA buffer for the drpai_buf\n");
        goto end_free_malloc;
    }

    std::cout << "[INFO] loaded runtime model :" << model_dir << "\n\n";
    switch (input_source_map[input_source])
    {
        
        case 1:
        {
            std::cout << "[INFO] USB CAMERA \n";
            status=query_device_status("usb");
            if(number_of_cameras>device_paths.size())  
            {
            cout<<"check no.of cameras connected and entered no.of cameras"<<endl;
            goto end_close_drpai;
            }  

            if(status ==1)
            {
                for(i=0;i<number_of_cameras;i++)
                {
                gstreamer_pipeline.push_back("v4l2src device=" + device_paths[i] + " ! video/x-raw, width=640, height=480, framerate=10/1 ! videoconvert ! appsink -v");
                }
                /* Initialize waylad */
                ret = wayland.init(DISP_OUTPUT_WIDTH, DISP_OUTPUT_HEIGHT, BGRA_CHANNEL);
                if(0 != ret)
                {
                    fprintf(stderr, "[ERROR] Failed to initialize Image for Wayland\n");
                    ret_main = -1;
                    goto end_close_drpai;
                }

                /*Termination Request Semaphore Initialization*/
                /*Initialized value at 1.*/
                sem_create = sem_init(&terminate_req_sem, 0, 1);
                if (0 != sem_create)
                {
                    fprintf(stderr, "[ERROR] Failed to Initialize Termination Request Semaphore.\n");
                    ret_main = -1;
                    goto end_threads;
                }
                /*Create Key Hit Thread*/
                create_thread_key = pthread_create(&kbhit_thread, NULL, R_Kbhit_Thread, NULL);
                if (0 != create_thread_key)
                {
                    fprintf(stderr, "[ERROR] Failed to create Key Hit Thread.\n");
                    ret_main = -1;
                    goto end_threads;
                }
                /*Create Inference Thread*/
                create_thread_ai = pthread_create(&ai_inf_thread, NULL, R_Inf_Thread, NULL);
                if (0 != create_thread_ai)
                {
                    sem_trywait(&terminate_req_sem);
                    fprintf(stderr, "[ERROR] Failed to create AI Inference Thread.\n");
                    ret_main = -1;
                    goto end_threads;
                }
                /*Create Capture Thread*/
                create_thread_capture = pthread_create(&capture_thread, NULL, R_Capture_Thread, NULL);
                if (0 != create_thread_capture)
                {
                    sem_trywait(&terminate_req_sem);
                    fprintf(stderr, "[ERROR] Failed to create Capture Thread.\n");
                    ret_main = -1;
                    goto end_threads;
                }
                /*Create Memory Copy Thread*/
                create_thread_memcpy = pthread_create(&memcpy_thread, NULL, R_Memcpy_Thread, NULL);
                if (0 != create_thread_memcpy)
                {
                    sem_trywait(&terminate_req_sem);
                    fprintf(stderr, "[ERROR] Failed to create Memory Copy Thread.\n");
                    ret_main = -1;
                    goto end_threads;
                }
            }
            else
            {
                cout<<"unable to get device paths"<<endl;
                goto end_close_drpai;
            }
        }
        break;
        case 2:
        {
            std::cout << "[INFO] MIPI CAMERA \n";
            status=query_device_status("CRU");
            if(number_of_cameras>device_paths.size())  
            {
            cout<<"check no.of cameras connected and entered no.of cameras"<<endl;
            goto end_close_drpai;
            }  
        
            if(status ==1)
            {
                for(i=0;i<number_of_cameras;i++)
                {
                gstreamer_pipeline.push_back("v4l2src device=" + device_paths[i] + " ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! appsink -v");
                }
                #ifdef V2N
                    if(number_of_cameras==1)
                    {
                        std::string sw_cmd1 = format("media-ctl -d /dev/media0 -V \"'csi-16000400.csi20':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd2 = format("media-ctl -d /dev/media0 -V \"'imx462 0-001f':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd3 = format("media-ctl -d /dev/media0 -V \"'cru-ip-16000000.cru0':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd4 = format("media-ctl -d /dev/media0 -V \"'cru-ip-16000000.cru0':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        const char* commands[9] =
                        {
                            "v4l2-ctl -d 0 -c framerate=30",
                            "v4l2-ctl -d 0 -c white_balance_auto_preset=0",
                            "media-ctl -d /dev/media0 -r",
                            "media-ctl -d /dev/media0 -l \"'csi-16000400.csi20':1 -> 'cru-ip-16000000.cru0':0 [1]\"",
                            "media-ctl -d /dev/media0 -l \"'cru-ip-16000000.cru0':1 -> 'CRU output':0 [1]\"",
                            sw_cmd1.c_str(),
                            sw_cmd2.c_str(),
                            sw_cmd3.c_str(),
                            sw_cmd4.c_str(),
                        };
                        int cmd_count = 9;
                        for (i = 0; i < cmd_count; i++)
                        {
                            ret = system(commands[i]);
                            if (ret < 0)
                            {
                                printf("%s: failed media-ctl commands. index = %d\n", __func__, i);
                                return -1;
                            }
                        }
                    }
                    if(number_of_cameras==2)
                    {
                        std::string sw_cmd1 = format("media-ctl -d /dev/media0 -V \"'csi-16000400.csi20':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd2 = format("media-ctl -d /dev/media0 -V \"'imx462 0-001f':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd3 = format("media-ctl -d /dev/media0 -V \"'cru-ip-16000000.cru0':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd4 = format("media-ctl -d /dev/media0 -V \"'cru-ip-16000000.cru0':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        const char* commands[9] =
                        {
                            "v4l2-ctl -d 0 -c framerate=30",
                            "v4l2-ctl -d 0 -c white_balance_auto_preset=0",
                            "media-ctl -d /dev/media0 -r",
                            "media-ctl -d /dev/media0 -l \"'csi-16000400.csi20':1 -> 'cru-ip-16000000.cru0':0 [1]\"",
                            "media-ctl -d /dev/media0 -l \"'cru-ip-16000000.cru0':1 -> 'CRU output':0 [1]\"",
                            sw_cmd1.c_str(),
                            sw_cmd2.c_str(),
                            sw_cmd3.c_str(),
                            sw_cmd4.c_str(),
                        };
                        int cmd_count = 9;
                        for (i = 0; i < cmd_count; i++)
                        {
                            ret = system(commands[i]);
                            if (ret < 0)
                            {
                                printf("%s: failed media-ctl commands. index = %d\n", __func__, i);
                                return -1;
                            }
                        }

                        std::string sw_cmd5 = format("media-ctl -d /dev/media1 -V \"'csi-16010400.csi21':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd6 = format("media-ctl -d /dev/media1 -V \"'imx462 1-001f':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd7 = format("media-ctl -d /dev/media1 -V \"'cru-ip-16010000.cru1':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd8 = format("media-ctl -d /dev/media1 -V \"'cru-ip-16010000.cru1':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        const char* command[9] =
                        {
                            "v4l2-ctl -d 1 -c framerate=30",
                            "v4l2-ctl -d 1 -c white_balance_auto_preset=0",
                            "media-ctl -d /dev/media1 -r",
                            "media-ctl -d /dev/media1 -l \"'csi-16010400.csi21':1 -> 'cru-ip-16010000.cru1':0 [1]\"",
                            "media-ctl -d /dev/media1 -l \"'cru-ip-16010000.cru1':1 -> 'CRU output':0 [1]\"",
                            sw_cmd5.c_str(),
                            sw_cmd6.c_str(),
                            sw_cmd7.c_str(),
                            sw_cmd8.c_str(),
                        };
                        cmd_count = 9;
                        for (i = 0; i < cmd_count; i++)
                        {
                            ret = system(command[i]);
                            if (ret < 0)
                            {
                                printf("%s: failed media-ctl commands. index = %d\n", __func__, i);
                                return -1;
                            }
                        }
                    
                    }

                    if(number_of_cameras==3)
                    {
                        std::string sw_cmd1 = format("media-ctl -d /dev/media0 -V \"'csi-16000400.csi20':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd2 = format("media-ctl -d /dev/media0 -V \"'imx462 0-001f':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd3 = format("media-ctl -d /dev/media0 -V \"'cru-ip-16000000.cru0':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd4 = format("media-ctl -d /dev/media0 -V \"'cru-ip-16000000.cru0':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        const char* commands[9] =
                        {
                            "v4l2-ctl -d 0 -c framerate=30",
                            "v4l2-ctl -d 0 -c white_balance_auto_preset=0",
                            "media-ctl -d /dev/media0 -r",
                            "media-ctl -d /dev/media0 -l \"'csi-16000400.csi20':1 -> 'cru-ip-16000000.cru0':0 [1]\"",
                            "media-ctl -d /dev/media0 -l \"'cru-ip-16000000.cru0':1 -> 'CRU output':0 [1]\"",
                            sw_cmd1.c_str(),
                            sw_cmd2.c_str(),
                            sw_cmd3.c_str(),
                            sw_cmd4.c_str(),
                        };
                        int cmd_count = 9;
                        for (i = 0; i < cmd_count; i++)
                        {
                            ret = system(commands[i]);
                            if (ret < 0)
                            {
                                printf("%s: failed media-ctl commands. index = %d\n", __func__, i);
                                return -1;
                            }
                        }

                        std::string sw_cmd5 = format("media-ctl -d /dev/media1 -V \"'csi-16010400.csi21':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd6 = format("media-ctl -d /dev/media1 -V \"'imx462 1-001f':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd7 = format("media-ctl -d /dev/media1 -V \"'cru-ip-16010000.cru1':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd8 = format("media-ctl -d /dev/media1 -V \"'cru-ip-16010000.cru1':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        const char* command[9] =
                        {
                            "v4l2-ctl -d 1 -c framerate=30",
                            "v4l2-ctl -d 1 -c white_balance_auto_preset=0",
                            "media-ctl -d /dev/media1 -r",
                            "media-ctl -d /dev/media1 -l \"'csi-16010400.csi21':1 -> 'cru-ip-16010000.cru1':0 [1]\"",
                            "media-ctl -d /dev/media1 -l \"'cru-ip-16010000.cru1':1 -> 'CRU output':0 [1]\"",
                            sw_cmd5.c_str(),
                            sw_cmd6.c_str(),
                            sw_cmd7.c_str(),
                            sw_cmd8.c_str(),
                        };
                        cmd_count = 9;
                        for (i = 0; i < cmd_count; i++)
                        {
                            ret = system(command[i]);
                            if (ret < 0)
                            {
                                printf("%s: failed media-ctl commands. index = %d\n", __func__, i);
                                return -1;
                            }
                        }

                        std::string sw_cmd9 = format("media-ctl -d /dev/media2 -V \"'csi-16020400.csi22':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd10 = format("media-ctl -d /dev/media2 -V \"'imx462 2-001f':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd11 = format("media-ctl -d /dev/media2 -V \"'cru-ip-16020000.cru2':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd12 = format("media-ctl -d /dev/media2 -V \"'cru-ip-16020000.cru2':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        const char* command_third[9] =
                        {
                            "v4l2-ctl -d 2 -c framerate=30",
                            "v4l2-ctl -d 2 -c white_balance_auto_preset=0",
                            "media-ctl -d /dev/media2 -r",
                            "media-ctl -d /dev/media2 -l \"'csi-16020400.csi22':1 -> 'cru-ip-16020000.cru2':0 [1]\"",
                            "media-ctl -d /dev/media2 -l \"'cru-ip-16020000.cru2':1 -> 'CRU output':0 [1]\"",
                            sw_cmd9.c_str(),
                            sw_cmd10.c_str(),
                            sw_cmd11.c_str(),
                            sw_cmd12.c_str(),
                        };
                        cmd_count = 9;
                        for (i = 0; i < cmd_count; i++)
                        {
                            ret = system(command_third[i]);
                            if (ret < 0)
                            {
                                printf("%s: failed media-ctl commands. index = %d\n", __func__, i);
                                return -1;
                            }
                        }
                    
                    }
                    if(number_of_cameras==4)
                    {
                        std::string sw_cmd1 = format("media-ctl -d /dev/media0 -V \"'csi-16000400.csi20':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd2 = format("media-ctl -d /dev/media0 -V \"'imx462 0-001f':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd3 = format("media-ctl -d /dev/media0 -V \"'cru-ip-16000000.cru0':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd4 = format("media-ctl -d /dev/media0 -V \"'cru-ip-16000000.cru0':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        const char* commands[9] =
                        {
                            "v4l2-ctl -d 0 -c framerate=30",
                            "v4l2-ctl -d 0 -c white_balance_auto_preset=0",
                            "media-ctl -d /dev/media0 -r",
                            "media-ctl -d /dev/media0 -l \"'csi-16000400.csi20':1 -> 'cru-ip-16000000.cru0':0 [1]\"",
                            "media-ctl -d /dev/media0 -l \"'cru-ip-16000000.cru0':1 -> 'CRU output':0 [1]\"",
                            sw_cmd1.c_str(),
                            sw_cmd2.c_str(),
                            sw_cmd3.c_str(),
                            sw_cmd4.c_str(),
                        };
                        int cmd_count = 9;
                        for (i = 0; i < cmd_count; i++)
                        {
                            ret = system(commands[i]);
                            if (ret < 0)
                            {
                                printf("%s: failed media-ctl commands. index = %d\n", __func__, i);
                                return -1;
                            }
                        }

                        std::string sw_cmd5 = format("media-ctl -d /dev/media1 -V \"'csi-16010400.csi21':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd6 = format("media-ctl -d /dev/media1 -V \"'imx462 1-001f':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd7 = format("media-ctl -d /dev/media1 -V \"'cru-ip-16010000.cru1':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd8 = format("media-ctl -d /dev/media1 -V \"'cru-ip-16010000.cru1':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        const char* command[9] =
                        {
                            "v4l2-ctl -d 1 -c framerate=30",
                            "v4l2-ctl -d 1 -c white_balance_auto_preset=0",
                            "media-ctl -d /dev/media1 -r",
                            "media-ctl -d /dev/media1 -l \"'csi-16010400.csi21':1 -> 'cru-ip-16010000.cru1':0 [1]\"",
                            "media-ctl -d /dev/media1 -l \"'cru-ip-16010000.cru1':1 -> 'CRU output':0 [1]\"",
                            sw_cmd5.c_str(),
                            sw_cmd6.c_str(),
                            sw_cmd7.c_str(),
                            sw_cmd8.c_str(),
                        };
                        cmd_count = 9;
                        for (i = 0; i < cmd_count; i++)
                        {
                            ret = system(command[i]);
                            if (ret < 0)
                            {
                                printf("%s: failed media-ctl commands. index = %d\n", __func__, i);
                                return -1;
                            }
                        }
                        std::string sw_cmd9 = format("media-ctl -d /dev/media2 -V \"'csi-16020400.csi22':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd10 = format("media-ctl -d /dev/media2 -V \"'imx462 2-001f':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd11 = format("media-ctl -d /dev/media2 -V \"'cru-ip-16020000.cru2':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd12 = format("media-ctl -d /dev/media2 -V \"'cru-ip-16020000.cru2':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        const char* command_third[9] =
                        {
                            "v4l2-ctl -d 2 -c framerate=30",
                            "v4l2-ctl -d 2 -c white_balance_auto_preset=0",
                            "media-ctl -d /dev/media2 -r",
                            "media-ctl -d /dev/media2 -l \"'csi-16020400.csi22':1 -> 'cru-ip-16020000.cru2':0 [1]\"",
                            "media-ctl -d /dev/media2 -l \"'cru-ip-16020000.cru2':1 -> 'CRU output':0 [1]\"",
                            sw_cmd9.c_str(),
                            sw_cmd10.c_str(),
                            sw_cmd11.c_str(),
                            sw_cmd12.c_str(),
                        };
                        cmd_count = 9;
                        for (i = 0; i < cmd_count; i++)
                        {
                            ret = system(command_third[i]);
                            if (ret < 0)
                            {
                                printf("%s: failed media-ctl commands. index = %d\n", __func__, i);
                                return -1;
                            }
                        }

                        std::string sw_cmd13 = format("media-ctl -d /dev/media3 -V \"'csi-16030400.csi23':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd14 = format("media-ctl -d /dev/media3 -V \"'imx462 3-001f':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd15 = format("media-ctl -d /dev/media3 -V \"'cru-ip-16030000.cru3':0 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        std::string sw_cmd16 = format("media-ctl -d /dev/media3 -V \"'cru-ip-16030000.cru3':1 [fmt:UYVY8_2X8/%s field:none]\"", MIPI_CAM_RES);
                        const char* command_fourth[9] =
                        {
                            "v4l2-ctl -d 3 -c framerate=30",
                            "v4l2-ctl -d 3 -c white_balance_auto_preset=0",
                            "media-ctl -d /dev/media3 -r",
                            "media-ctl -d /dev/media3 -l \"'csi-16030400.csi23':1 -> 'cru-ip-16030000.cru3':0 [1]\"",
                            "media-ctl -d /dev/media3 -l \"'cru-ip-16030000.cru3':1 -> 'CRU output':0 [1]\"",
                            sw_cmd13.c_str(),
                            sw_cmd14.c_str(),
                            sw_cmd15.c_str(),
                            sw_cmd16.c_str(),
                        };
                        cmd_count = 9;
                        for (i = 0; i < cmd_count; i++)
                        {
                            ret = system(command_fourth[i]);
                            if (ret < 0)
                            {
                                printf("%s: failed media-ctl commands. index = %d\n", __func__, i);
                                return -1;
                            }
                        }
                    
                    }
                #else  // not V2N
                    for(i=0;i<number_of_cameras;i++)
                    {
                        white_space="";
                        //removal of white space from string 
                        for (char ch : device_paths[i]) 
                        {
                            if (!isspace(ch)) 
                            {
                                white_space += ch;
                            }
                        }

                        command = "v4l2-ctl -d " + white_space + " -c framerate=30";
                        std::system(command.c_str());

                        command = "v4l2-ctl -d " + white_space + " -c white_balance_auto_preset=0";
                        std::system(command.c_str());

                        command = "media-ctl -d /dev/media" + std::to_string(i) + " -r";
                        std::system(command.c_str());

                        command = "media-ctl -d /dev/media" + std::to_string(i) + " -l \"\'rzg2l_csi2 160" + std::to_string(i) + "0400.csi2" + std::to_string(i) + "\':1 -> \'CRU output\':0 [1]\"";       
                        std::system(command.c_str());
                        
                        command = "media-ctl -d /dev/media" + std::to_string(i) + " -V \"\'rzg2l_csi2 160" + std::to_string(i) + "0400.csi2" + std::to_string(i) + "\':1 [fmt:UYVY8_2X8/640x480 field:none]\"";             
                        std::system(command.c_str());

                        if(i==0 || i==1)
                        {
                        command = "media-ctl -d /dev/media" + std::to_string(i) + " -V \"\'imx462 " + std::to_string(i) + "-001f\':0 [fmt:UYVY8_2X8/640x480 field:none]\"";     
                        std::system(command.c_str());
                        }
                        else if(i==2)
                        {
                        command = "media-ctl -d /dev/media" + std::to_string(i) + " -V \"\'imx462 " + std::to_string(x) + "-001f\':0 [fmt:UYVY8_2X8/640x480 field:none]\"";     
                        std::system(command.c_str());
                        }
                        else
                        {
                        command = "media-ctl -d /dev/media" + std::to_string(i) + " -V \"\'imx462 " + std::to_string(y) + "-001f\':0 [fmt:UYVY8_2X8/640x480 field:none]\"";     
                        std::system(command.c_str());
                        }                    
                    }
                #endif
                    /* Initialize waylad */
                    ret = wayland.init(DISP_OUTPUT_WIDTH, DISP_OUTPUT_HEIGHT, BGRA_CHANNEL);
                    if(0 != ret)
                    {
                        fprintf(stderr, "[ERROR] Failed to initialize Image for Wayland\n");
                        ret_main = -1;
                        goto end_close_drpai;
                    }

                    /*Termination Request Semaphore Initialization*/
                    /*Initialized value at 1.*/
                    sem_create = sem_init(&terminate_req_sem, 0, 1);
                    if (0 != sem_create)
                    {
                        fprintf(stderr, "[ERROR] Failed to Initialize Termination Request Semaphore.\n");
                        ret_main = -1;
                        goto end_threads;
                    }
                    /*Create Key Hit Thread*/
                    create_thread_key = pthread_create(&kbhit_thread, NULL, R_Kbhit_Thread, NULL);
                    if (0 != create_thread_key)
                    {
                        fprintf(stderr, "[ERROR] Failed to create Key Hit Thread.\n");
                        ret_main = -1;
                        goto end_threads;
                    }
                    /*Create Inference Thread*/
                    create_thread_ai = pthread_create(&ai_inf_thread, NULL, R_Inf_Thread, NULL);
                    if (0 != create_thread_ai)
                    {
                        sem_trywait(&terminate_req_sem);
                        fprintf(stderr, "[ERROR] Failed to create AI Inference Thread.\n");
                        ret_main = -1;
                        goto end_threads;
                    }
                    /*Create Capture Thread*/
                    create_thread_capture = pthread_create(&capture_thread, NULL, R_Capture_Thread, NULL);
                    if (0 != create_thread_capture)
                    {
                        sem_trywait(&terminate_req_sem);
                        fprintf(stderr, "[ERROR] Failed to create Capture Thread.\n");
                        ret_main = -1;
                        goto end_threads;
                    }
                    /*Create Memory Copy Thread*/
                    create_thread_memcpy = pthread_create(&memcpy_thread, NULL, R_Memcpy_Thread, NULL);
                    if (0 != create_thread_memcpy)
                    {
                        sem_trywait(&terminate_req_sem);
                        fprintf(stderr, "[ERROR] Failed to create Memory Copy Thread.\n");
                        ret_main = -1;
                        goto end_threads;
                    }
            }
            else
            {
                cout<<"unable to get device paths"<<endl;
                goto end_close_drpai;
            }
            break;
        }
    }

    /*Main Processing*/
    main_proc = R_Main_Process();
    if (0 != main_proc)
    {
        fprintf(stderr, "[ERROR] Error during Main Process\n");
        ret_main = -1;
    }
    goto end_threads;

end_threads:
    if(0 == create_thread_capture)
    {
        ret = wait_join(&capture_thread, CAPTURE_TIMEOUT);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to exit Capture Thread on time.\n");
            ret_main = -1;
        }
    }
    if (0 == create_thread_memcpy)
    {
        ret = wait_join(&memcpy_thread, CAPTURE_TIMEOUT);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to exit Memory Copy Thread on time.\n");
            ret_main = -1;
        }
    }
    if (0 == create_thread_ai)
    {
        ret = wait_join(&ai_inf_thread, AI_THREAD_TIMEOUT);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to exit AI Inference Thread on time.\n");
            ret_main = -1;
        }
    }
    if (0 == create_thread_key)
    {
        ret = wait_join(&kbhit_thread, KEY_THREAD_TIMEOUT);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to exit Key Hit Thread on time.\n");
            ret_main = -1;
        }
    }

    /*Delete Terminate Request Semaphore.*/
    if (0 == sem_create)
    {
        sem_destroy(&terminate_req_sem);
    }

    /* Exit the program */
    wayland.exit();

    goto end_free_malloc;
end_free_malloc:
    free(drpai_buf);
    free(drpai_buf1);
    free(drpai_buf2);
    free(drpai_buf3);
    drpai_buf = NULL;
    drpai_buf1 = NULL;
    drpai_buf2 = NULL;
    drpai_buf3 = NULL;

    goto end_close_drpai;

end_close_drpai:
    /*Close DRP-AI Driver.*/
    if (0 < drpai_fd)
    {
        errno = 0;
        ret = close(drpai_fd);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to close DRP-AI Driver: errno=%d\n", errno);
            ret_main = -1;
        }
    }
    goto end_main;

end_main:
    printf("Application End\n");
    return ret_main;
}
