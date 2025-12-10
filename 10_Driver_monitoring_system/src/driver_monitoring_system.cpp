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
* File Name    : driver_monitoring_system.cpp
* Version      : 1.0.0
* Description  : DRP-AI Driver monitoring system detection application
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
/* Tracker logic header files */
#include <deque>
#include <map>
#include <algorithm>
static std:: deque<int> head_pose_history; // store last 5 YOLO head-pose class IDs
static const int HEAD_POSE_WINDOW = 5;      // 5-frame averaging
std::string cam_angle;  // Camera angle

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

/*Global Variables*/
static float drpai_output_buf[INF_OUT_SIZE];
static Wayland wayland;
static pthread_t ai_inf_thread;
static pthread_t kbhit_thread;
static pthread_t capture_thread;
static pthread_t memcpy_thread;
static mutex mtx;

/*Flags*/
static std::atomic<uint8_t> capture_start           (1);
static std::atomic<uint8_t> inference_start         (0);
static std::atomic<uint8_t> img_processing_start    (0);

/* Flag set when user requests termination (Enter key). */
/* Used to distinguish normal user-initiated shutdown from error conditions. */
static std::atomic<bool> termination_requested(false);

float sum_inf_time = 0.0;
float sum_pre_poc_time = 0.0;
float sum_post_proc_time = 0.0;
float sum_total_time = 0.0;
static int inference_frame_counter = 0;
static auto last_inf_time = std::chrono::high_resolution_clock::now();
float camera_fps = 0.0f;

static sem_t terminate_req_sem;
static int32_t drpai_freq;
static int32_t number_of_cameras;
std::vector<std::string> gstreamer_pipeline;
std::vector<std::string> device_paths;
std::vector<VideoCapture> cap;

uint64_t drpaimem_addr_start = 0;
bool runtime_status = false; 
static vector<detection> det;

/*Global frame */
std::vector<Mat> g_frame;
std::vector<Mat> cap_frame;
std::vector<Mat> inf_frames=
{
   cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0))
};
std::vector<Mat> img_frames=
{
    cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0))
};

/* Map to store input source list */
std::map<std::string, int> input_source_map ={    
    {"USB", 1},
    {"MIPI", 2}
    };


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
* Description   : Process CPU post-processing for yolox-l
* Arguments     : floatarr = drpai output address
* Return value  : -
******************************************/
void R_Post_Proc(float* floatarr)
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
    det.clear();
    copy(det_buff.begin(), det_buff.end(), back_inserter(det));
    mtx.unlock();
    return ;
}

/*****************************************
 * Function Name : head_class_tracker
 * Description   : Tracker function to stabilize head class
 * Arguments     : - new_cls (current raw prediction)
 * Return value  : - final (smoothed valid pose)
 ******************************************/
int head_class_tracker(int new_cls)
{
    const int HEAD_LEFT = 4;
    const int HEAD_RIGHT = 5;
    const int HEAD_CENTER = 2;
    const int HEAD_DOWN = 3;

    // Only manage head pose classes 2–6
    if (new_cls < HEAD_CENTER || new_cls > HEAD_RIGHT)
        return new_cls;

    // Get last stable pose if any
    int last_pose = head_pose_history.empty() ? new_cls : head_pose_history.back();

    // ---------- REJECT SUDDEN OPPOSITE SWITCH ----------
    // RULE: Left cannot jump to RIGHT without CENTER/DOWN in-between
    if ((last_pose == HEAD_LEFT  && new_cls == HEAD_RIGHT) ||
        (last_pose == HEAD_RIGHT && new_cls == HEAD_LEFT))
    {
        // Check last 5 frames – did we pass CENTER or DOWN?
        bool passed_center = false;
        for (int c : head_pose_history)
            if (c == HEAD_CENTER || c == HEAD_DOWN)
                passed_center = true;

        if (!passed_center)
        {
            // Reject and return last pose
            new_cls = last_pose;
        }
    }

    // ---------- Push to history ----------
    head_pose_history.push_back(new_cls);
    if (head_pose_history.size() > HEAD_POSE_WINDOW)
        head_pose_history.pop_front();

    // ---------- Compute Mode ----------
    std::map<int,int> freq;
    for (int c : head_pose_history)
        freq[c]++;

    int best = new_cls, max_count = 0;
    for (auto &kv : freq)
    {
        if (kv.second > max_count)
        {
            max_count = kv.second;
            best = kv.first;
        }
    }

    int final = best;

    // ---------- CAMERA ANGLE CORRECTION ----------
    if (cam_angle == "LEFT")
    {
        if (final == HEAD_LEFT)  final = HEAD_RIGHT;
        else if (final == HEAD_RIGHT) final = HEAD_LEFT;
    }
    return final;
}

/*****************************************
 * Function Name : suppress_head_pose_duplicates
 * Description   : Remove duplicate head-pose predictions (head pose classes 2–6)
 * Arguments     : - 
 * Return value  : -  single head pose class with high confidence
 ******************************************/
void suppress_head_pose_duplicates(std::vector<detection>& boxes)
{
    const int HEAD_POSE_MIN = 2;
    const int HEAD_POSE_MAX = 6;

    detection best_box;
    bool found = false;

    for (auto& d : boxes)
    {
        if (d.c >= HEAD_POSE_MIN && d.c <= HEAD_POSE_MAX)
        {
            if (!found || d.prob > best_box.prob)
            {
                best_box = d;
                found = true;
            }
        }
    }

    if (!found) return;

    std::vector<detection> filtered;
    for (auto& d : boxes)
    {
        if (d.c >= HEAD_POSE_MIN && d.c <= HEAD_POSE_MAX)
        {
            if (d.bbox.x == best_box.bbox.x &&
                d.bbox.y == best_box.bbox.y)
            {
                filtered.push_back(best_box);
            }
        }
        else
        {
            filtered.push_back(d);
        }
    }
    boxes = filtered;
}

/***************************************** 
 * Function Name : draw_bounding_box 
 * Description : Draw bounding box on image. 
 * Arguments : - 
 * Return value : - 
******************************************/ 
void draw_bounding_box(void) 
{ 
    vector<detection> tmp_buff; 
    { 
        copy(det.begin(), det.end(), back_inserter(tmp_buff)); 
    } 
    if (tmp_buff.empty()) 
        return; 
    suppress_head_pose_duplicates(tmp_buff); 
    stringstream stream; 
    string str; 
    for (size_t i = 0; i < tmp_buff.size(); i++) 
    { 
        int updated_class = head_class_tracker(tmp_buff[i].c); 
        tmp_buff[i].c = updated_class; 
        if(tmp_buff[i].c == 9) 
        { 
            stream.str(""); 
            stream << "Yawn Detected"; 
            str = stream.str(); 
            putText(g_frame[0], str, Point(YAWN_STR_X, YAWN_STR_Y), FONT_HERSHEY_SIMPLEX, DMS_CHAR_SCALE_SMALL, Scalar(0,0,0), 2); 
            putText(g_frame[0], str, Point(YAWN_STR_X, YAWN_STR_Y), FONT_HERSHEY_SIMPLEX, DMS_CHAR_SCALE_SMALL, Scalar(0,255,255), 1); 
        } 
        if(tmp_buff[i].c == 0) 
        { 
            stream.str(""); 
            stream << "Blink Detected"; 
            str = stream.str(); 
            putText(g_frame[0], str, Point(BLINK_STR_X, BLINK_STR_Y), FONT_HERSHEY_SIMPLEX, DMS_CHAR_SCALE_SMALL, Scalar(0,0,0), 2); 
            putText(g_frame[0], str, Point(BLINK_STR_X, BLINK_STR_Y), FONT_HERSHEY_SIMPLEX, DMS_CHAR_SCALE_SMALL, Scalar(0,255,255), 1); 
        } 
        if (tmp_buff[i].c >= 2 && tmp_buff[i].c <= 5) 
        { 
            stream.str(""); stream << "Head Pose: " << label_file_map[tmp_buff[i].c]; 
            str = stream.str(); 
            putText(g_frame[0], str, Point(HEAD_POSE_STR_X, HEAD_POSE_STR_Y), FONT_HERSHEY_SIMPLEX, DMS_CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5 * DMS_CHAR_THICKNESS); 
            putText(g_frame[0], str, Point(HEAD_POSE_STR_X, HEAD_POSE_STR_Y), FONT_HERSHEY_SIMPLEX, DMS_CHAR_SCALE_SMALL, Scalar(0, 255, 255), DMS_CHAR_THICKNESS); 
        }
    } 
}

/*****************************************
 * Function Name : DMS_detection
 * Description   : Function to perform over all detection
 * Arguments     : -
 * Return value  : 0 if succeeded
 *                 not 0 otherwise
 ******************************************/
int DMS_detection()
{ 
    /*Variable for getting Inference output data*/
    void* output_ptr;
    uint32_t out_size;

    /*Variable for Pre-processing parameter configuration*/
    s_preproc_param_t in_param;

    /*Variable for checking return value*/
    int8_t ret = 0;
    /*load inference out on drpai_out_buffer*/
    int32_t i = 0;
    int32_t inf_num = 0;
    int32_t output_num = 0;
    std::tuple<InOutDataType, void *, int64_t> output_buffer;
    int64_t output_size;

    uint32_t size_count  = 0;
    
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

    R_Post_Proc(drpai_output_buf);
    auto t5 = std::chrono::high_resolution_clock::now();
    
    auto r_post_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
    float total_time = float(inf_duration/1000.0) + float(r_post_proc_time/1000.0) + float(pre_proc_time/1000.0); 

    sum_inf_time        = inf_duration  / 1000.0;
    sum_pre_poc_time    = pre_proc_time / 1000.0;
    sum_post_proc_time  = r_post_proc_time / 1000.0;
    sum_total_time      = total_time;

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
            // Mark that termination was requested by the user (Enter key)
            termination_requested.store(true);
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
            /*Checks if sem_getvalue is executed without issue*/
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
                /* Calculating the camera fps */
                camera_fps  = cap[i].get(CAP_PROP_FPS);
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
                
                memcpy( drpai_buf->mem,  input_images[0].data,  drpai_buf->size);

                /* Flush buffer */
                ret = buffer_flush_dmabuf(drpai_buf->idx, drpai_buf->size);
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

        int ret = DMS_detection();
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
    cv::Mat bgra_image   = cv::Mat(DISP_OUTPUT_HEIGHT,DISP_OUTPUT_WIDTH,CV_8UC4);
    uint8_t * img_buffer0;

    img_buffer0 = (unsigned char*)malloc(DISP_OUTPUT_WIDTH*DISP_OUTPUT_HEIGHT*BGRA_CHANNEL);

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

        /* Check if the captured frame is valid. This condition prevents processing 
        when the camera has not yet delivered a frame,or a frame was dropped/corrupted.*/
        if (g_frame.empty())
        {
            img_processing_start.store(0);
            continue;
        }

        /* Draw bounding box and Mosaic on the frame */
        draw_bounding_box();

        // Restore ORIGINAL image window size calculation
        Size size(DISP_INF_WIDTH, DISP_INF_HEIGHT);
        cv::Mat resized_frame;
        cv::resize(g_frame[0], resized_frame, size);

        // Copy into display region
        resized_frame.copyTo(output_image(Rect(FIRST_FRAME_X_COORDINATE, FIRST_FRAME_Y_COORDINATE,
                                                DISP_INF_WIDTH,DISP_INF_HEIGHT)));

        stream.str("");
        stream << "Total Time: " << fixed << setprecision(2)<< sum_total_time <<" ms";
        str = stream.str();
        Size tot_time_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_LARGE,
                                         HC_CHAR_THICKNESS, &baseline);
        putText(output_image, str,
                Point((DISP_OUTPUT_WIDTH - tot_time_size.width - RIGHT_ALIGN_OFFSET),
                      (T_TIME_STR_Y + tot_time_size.height)),
                FONT_HERSHEY_SIMPLEX, CHAR_SCALE_LARGE, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
        putText(output_image, str,
                Point((DISP_OUTPUT_WIDTH - tot_time_size.width - RIGHT_ALIGN_OFFSET),
                      (T_TIME_STR_Y + tot_time_size.height)),
                FONT_HERSHEY_SIMPLEX, CHAR_SCALE_LARGE, Scalar(0, 255, 0), HC_CHAR_THICKNESS);

        stream.str("");
        stream << "Yolox-l";
        str = stream.str();
        Size yoloxl_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL,
                                       HC_CHAR_THICKNESS, &baseline);
        putText(output_image, str,
                Point((DISP_OUTPUT_WIDTH - yoloxl_size.width - RIGHT_ALIGN_OFFSET),
                      (MODEL_NAME_1_Y + yoloxl_size.height)),
                FONT_HERSHEY_SIMPLEX, CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
        putText(output_image, str,
                Point((DISP_OUTPUT_WIDTH - yoloxl_size.width - RIGHT_ALIGN_OFFSET),
                      (MODEL_NAME_1_Y + yoloxl_size.height)),
                FONT_HERSHEY_SIMPLEX, CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);

        stream.str("");
        stream << "Pre-Proc: "  << fixed << setprecision(2)<< sum_pre_poc_time<<" ms";
        str = stream.str();
        Size pre_proc_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL,
                                         HC_CHAR_THICKNESS, &baseline);
        putText(output_image, str,
                Point((DISP_OUTPUT_WIDTH - pre_proc_size.width - RIGHT_ALIGN_OFFSET),
                      (PRE_TIME_STR_Y + pre_proc_size.height)),
                FONT_HERSHEY_SIMPLEX, CHAR_SCALE_SMALL,
                Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
        putText(output_image, str,
                Point((DISP_OUTPUT_WIDTH - pre_proc_size.width - RIGHT_ALIGN_OFFSET),
                      (PRE_TIME_STR_Y + pre_proc_size.height)),
                FONT_HERSHEY_SIMPLEX, CHAR_SCALE_SMALL,
                Scalar(255, 255, 255), HC_CHAR_THICKNESS);

        stream.str("");
        stream << "Inference: " << fixed << setprecision(2)<< sum_inf_time<<" ms";
        str = stream.str();
        Size inf_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL,
                                    HC_CHAR_THICKNESS, &baseline);
        putText(output_image, str,
                Point((DISP_OUTPUT_WIDTH - inf_size.width - RIGHT_ALIGN_OFFSET),
                      (I_TIME_STR_Y + inf_size.height)),
                FONT_HERSHEY_SIMPLEX, CHAR_SCALE_SMALL,
                Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
        putText(output_image, str,
                Point((DISP_OUTPUT_WIDTH - inf_size.width - RIGHT_ALIGN_OFFSET),
                      (I_TIME_STR_Y + inf_size.height)),
                FONT_HERSHEY_SIMPLEX, CHAR_SCALE_SMALL,
                Scalar(255, 255, 255), HC_CHAR_THICKNESS);

        stream.str("");
        stream << "Post-Proc: " << fixed << setprecision(2)<< sum_post_proc_time<<" ms";
        str = stream.str();
        Size post_proc_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL,
                                          HC_CHAR_THICKNESS, &baseline);
        putText(output_image, str,
                Point((DISP_OUTPUT_WIDTH - post_proc_size.width - RIGHT_ALIGN_OFFSET),
                      (P_TIME_STR_Y + post_proc_size.height)),
                FONT_HERSHEY_SIMPLEX, CHAR_SCALE_SMALL,
                Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
        putText(output_image, str,
                Point((DISP_OUTPUT_WIDTH - post_proc_size.width - RIGHT_ALIGN_OFFSET),
                      (P_TIME_STR_Y + post_proc_size.height)),
                FONT_HERSHEY_SIMPLEX, CHAR_SCALE_SMALL,
                Scalar(255, 255, 255), HC_CHAR_THICKNESS);

        stream.str("");
        stream << "Camera Frame Rate : "<< fixed << setprecision(1) << camera_fps <<" fps ";
        str = stream.str();
        Size camera_rate_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL,
                                            HC_CHAR_THICKNESS, &baseline);
        putText(output_image, str,
                Point((DISP_OUTPUT_WIDTH - camera_rate_size.width - RIGHT_ALIGN_OFFSET),
                      (FPS_STR_Y + camera_rate_size.height)),
                FONT_HERSHEY_SIMPLEX, CHAR_SCALE_SMALL,
                Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
        putText(output_image, str,
                Point((DISP_OUTPUT_WIDTH - camera_rate_size.width - RIGHT_ALIGN_OFFSET),
                      (FPS_STR_Y + camera_rate_size.height)),
                FONT_HERSHEY_SIMPLEX, CHAR_SCALE_SMALL,
                Scalar(255, 255, 255), HC_CHAR_THICKNESS);

        // Convert to BGRA & display
        cv::cvtColor(output_image, bgra_image, cv::COLOR_BGR2BGRA);
        memcpy(img_buffer0, bgra_image.data, DISP_OUTPUT_WIDTH * DISP_OUTPUT_HEIGHT * BGRA_CHANNEL);
        wayland.commit(img_buffer0, NULL);

        img_processing_start.store(0);
    }

    goto main_proc_end;

err:
    sem_trywait(&terminate_req_sem);
    /* If termination was explicitly requested by the user, treat as normal exit. */
    if (termination_requested.load())
    {
        main_ret = 0; // normal termination
    }
    else
    {
        main_ret = 1; // error
    }
    goto main_proc_end;

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
    std::string command;
    int32_t i,status = 0;
    int x=6;
    int y=7;
    std::string input_source = argv[1];
    std::cout << "Starting Driver Monitoring System Application" << std::endl;

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

    /*Disable OpenCV Accelerator due to the use of multithreading */
    #ifdef V2N
    unsigned long OCA_list[16];
    for (int i=0; i < 16; i++)
    {
        OCA_list[i] = 0;
    }
    #else /*for V2H*/
    unsigned long OCA_list[OCA_LIST_NUM];
    for (int i=0; i < OCA_LIST_NUM; i++)
    {
        OCA_list[i] = 0;
    }
    #endif
    OCA_Activate( &OCA_list[0] );

    if (strcmp(argv[1],"USB")==0 || strcmp(argv[1],"MIPI")==0)
    {  
        if(argc == 2)
        {
            cout << "Enter camera's orientation RIGHT/LEFT" << endl;
            goto end_close_drpai;
        }
        else if(argc == 3)
        {                      
            if(strcmp(argv[1],"USB")==0)
            {
                if(strcmp(argv[2],"RIGHT")==0 || strcmp(argv[2],"LEFT")==0)
                {
                    number_of_cameras = 1;
                    cam_angle = argv[2];
                    drpai_freq = DRPAI_FREQ;
                }
                else
                {
                    cout << "Enter camera's argument as LEFT/RIGHT for camera orientation" << endl;
                    goto end_close_drpai;
                }
            }
            else   // MIPI case
            {
                if(strcmp(argv[2],"RIGHT")==0 || strcmp(argv[2],"LEFT")==0)
                {
                    number_of_cameras = 1;
                    cam_angle = argv[2];
                    drpai_freq = DRPAI_FREQ;
                }
                else
                {
                    cout << "Enter camera's orientation as LEFT/RIGHT " << endl;
                    goto end_close_drpai;
                }
            }
        }
        else if (argc == 4)
        {
            if((atoi(argv[3]) <= (pow(2,sizeof(long long int) * 8 -1) -1)) && 
            (atoi(argv[3]) >= -(pow(2,sizeof(long long int) * 8 -1))))
            {
                drpai_freq = atoi(argv[3]);

                if ((1 <= drpai_freq) && (127 >= drpai_freq))
                {
                    if(strcmp(argv[1],"USB")==0)
                    {
                        if(strcmp(argv[2],"RIGHT")==0 || strcmp(argv[2],"LEFT")==0)
                        {
                            number_of_cameras = 1;
                            cam_angle = argv[2];
                        }
                        else
                        {
                            cout << "Enter camera's orientation as LEFT/RIGHT" << endl;
                            goto end_close_drpai;
                        }
                    }
                    else  // MIPI case
                    {
                        if(strcmp(argv[2],"RIGHT")==0 || strcmp(argv[2],"LEFT")==0)
                        {
                            number_of_cameras = 1;
                            cam_angle = argv[2];
                        }
                        else
                        {
                            cout << "Enter camera's orientation as LEFT/RIGHT" << endl;
                            goto end_close_drpai;
                        }
                    }
                }
                else
                {
                    cout << "Enter fourth argument as drp-ai frequency [1,127]" << endl; 
                    goto end_close_drpai;
                }

            }
            else
            {
                cout << "Enter integer in valid range" << endl;
                goto end_close_drpai;
            }
        }
        else
        {
            cout << "Wrong number of parameters passed" << endl;
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

    std::cout << "[INFO] loaded runtime model :" << model_dir << "\n\n";
    switch (input_source_map[input_source])
    {
        
        case 1:
        {
            std::cout << "[INFO] USB CAMERA \n";
            status=query_device_status("usb");
            if(status == 1)
            {
                for(i=0;i<number_of_cameras;i++)
                {
                gstreamer_pipeline.push_back("v4l2src device=" + device_paths[i] + " ! video/x-raw, width=640, height=480 ! videoconvert ! appsink -v");
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
            if(status == 1)
            {
                for(i=0;i<number_of_cameras;i++)
                {
                gstreamer_pipeline.push_back("v4l2src device=" + device_paths[i] + " ! video/x-raw, format=YUY2, width=640, height=480, framerate=30/1 ! videoconvert ! appsink -v");
                }
                
                for(i=0;i<number_of_cameras;i++)
                {
                    command = "v4l2-ctl -d " + std::to_string(i) + " -c framerate=30";
                    std::system(command.c_str());
                    command = "v4l2-ctl -d " + std::to_string(i) + " -c white_balance_auto_preset=0";
                    std::system(command.c_str());
                    command = "media-ctl -d /dev/media" + std::to_string(i) + " -r";
                    std::system(command.c_str());
                    
                    command = "media-ctl -d /dev/media" + std::to_string(i) + " -V \"'csi-160" + std::to_string(i) + "0400.csi2" + std::to_string(i) + "':1 [fmt:UYVY8_2X8/640x480 field:none]\"";
                    std::system(command.c_str());
                    if(i==0 || i==1)
                    {
                        command = "media-ctl -d /dev/media" + std::to_string(i) + " -V \"'imx462 " + std::to_string(i) + "-001f':0 [fmt:UYVY8_2X8/640x480 field:none]\"";
                        std::system(command.c_str());
                    }
                    else if(i==2)
                    {
                        command = "media-ctl -d /dev/media" + std::to_string(i) + " -V \"'imx462 " + std::to_string(x) + "-001f':0 [fmt:UYVY8_2X8/640x480 field:none]\"";
                        std::system(command.c_str());

                    }
                    else
                    {
                        command = "media-ctl -d /dev/media" + std::to_string(i) + " -V \"'imx462 " + std::to_string(y) + "-001f':0 [fmt:UYVY8_2X8/640x480 field:none]\"";
                        std::system(command.c_str());
                    }
                    
                }
                for(i=0;i<number_of_cameras;i++)
                {
                    #ifdef V2N
                        command = "media-ctl -d /dev/media" + std::to_string(i) + " -l \"'csi-160" + std::to_string(i) + "0400.csi2" + std::to_string(i) + "':1 -> 'cru-ip-160" + std::to_string(i) + "0000.cru" + std::to_string(i) + "':0 [1]\"";
                        std::system(command.c_str());

                        command = "media-ctl -d /dev/media" + std::to_string(i) + " -l \"'cru-ip-160" + std::to_string(i) + "0000.cru" + std::to_string(i) + "':1 -> 'CRU output':0 [1]\"";
                        std::system(command.c_str());

                        command = "media-ctl -d /dev/media" + std::to_string(i) + " -V \"'cru-ip-160" + std::to_string(i) + "0000.cru" + std::to_string(i) + "':0 [fmt:UYVY8_2X8/640x480 field:none]\"";
                        std::system(command.c_str());

                        command = "media-ctl -d /dev/media" + std::to_string(i) + " -V \"'cru-ip-160" + std::to_string(i) + "0000.cru" + std::to_string(i) + "':1 [fmt:UYVY8_2X8/640x480 field:none]\"";
                        std::system(command.c_str());
                    #else  // V2H
                        command = "media-ctl -d /dev/media" + std::to_string(i) + " -l \"'csi-160" + std::to_string(i) + "0400.csi2" + std::to_string(i) + "':1 -> 'cru-ip-160" + std::to_string(i) + "0000.video" + std::to_string(i) + "':0 [1]\"";
                        std::system(command.c_str());

                        command = "media-ctl -d /dev/media" + std::to_string(i) + " -l \"'cru-ip-160" + std::to_string(i) + "0000.video" + std::to_string(i) + "':1 -> 'CRU output':0 [1]\"";
                        std::system(command.c_str());

                        command = "media-ctl -d /dev/media" + std::to_string(i) + " -V \"'cru-ip-160" + std::to_string(i) + "0000.video" + std::to_string(i) + "':0 [fmt:UYVY8_2X8/640x480 field:none]\"";
                        std::system(command.c_str());

                        command = "media-ctl -d /dev/media" + std::to_string(i) + " -V \"'cru-ip-160" + std::to_string(i) + "0000.video" + std::to_string(i) + "':1 [fmt:UYVY8_2X8/640x480 field:none]\"";
                        std::system(command.c_str());
                    #endif
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
    drpai_buf = NULL;

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