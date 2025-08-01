/*
 * Original Code (C) Copyright Edgecortix, Inc. 2022
 * Modified Code (C) Copyright Renesas Electronics Corporation 2023
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
* File Name    : Road_lane_segmentation.cpp
* Version      : 1.1.0
* Description  : DRP-AI TVM[*1] Application Example
***********************************************************************************************************************/

/*****************************************
 * includes
 *****************************************/
#include "define.h"
#include "MeraDrpRuntimeWrapper.h"
#include <linux/drpai.h>
#include <linux/input.h>
#include <builtin_fp16.h>
#include <opencv2/opencv.hpp>
#include "wayland.h"

using namespace std;
using namespace cv;

/* DRP-AI TVM[*1] Runtime object */
MeraDrpRuntimeWrapper runtime;

bool runtime_status = false;
uint64_t drpaimem_addr_start = 0;

std::string gstreamer_pipeline;
std::vector<float> floatarr(1);

/*Flags*/
static std::atomic<uint8_t> capture_start           (1);
static std::atomic<uint8_t> inference_start         (0);
static std::atomic<uint8_t> img_processing_start    (0);

cv::Mat g_frame;
static cv::Mat g_result_image;
cv::VideoCapture cap;

cv::Mat inf_frame(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
cv::Mat img_frame(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));

static Wayland wayland;
static pthread_t ai_inf_thread;
static pthread_t kbhit_thread;
static pthread_t capture_thread;
static pthread_t memcpy_thread;
static atomic<uint8_t> hdmi_obj_ready   (0);
static uint32_t disp_time = 0;
static sem_t terminate_req_sem;

static std::mutex mtx;
static std::mutex mtx1;

static int32_t drpai_freq;

float TOTAL_TIME = 0;
float INF_TIME= 0;
float POST_PROC_TIME = 0;
float PRE_PROC_TIME = 0;


/* Map to store input source list */
std::map<std::string, int> input_source_map =
{
    {"USB", 1}
};

/*****************************************
 * Function Name     : float16_to_float32
 * Description       : Function by Edge cortex. Cast uint16_t a into float value.
 * Arguments         : a = uint16_t number
 * Return value      : float = float32 number
 ******************************************/
float float16_to_float32(uint16_t a)
{
    return __extendXfYf2__<uint16_t, uint16_t, 10, float, uint32_t, 23>(a);
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
 * Function Name : colour_convert
 * Description   : function to convert white colour to green colour.
 * Arguments     : Mat image
 * Return value  : Mat result
 ******************************************/
cv::Mat colour_convert(cv::Mat image)
{
    /* Convert the image to HSV */ 
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    /* Define the lower and upper HSV range for white color */
    cv::Scalar lower_white = cv::Scalar(0, 0, 200); // Adjust these values as needed
    cv::Scalar upper_white = cv::Scalar(180, 30, 255); // Adjust these values as needed

    /* Create a mask for the white color */
    cv::Mat mask;
    cv::inRange(hsv, lower_white, upper_white, mask);

    /* Create a green image */
    cv::Mat green_image = cv::Mat::zeros(image.size(), image.type());
    green_image.setTo(cv::Scalar(0, 255, 0), mask);

    /* Replace white regions in the original image with green */
    cv::Mat result;
    cv::bitwise_and(image, image, result, ~mask);
    cv::add(result, green_image, result);
    cv::resize(result, result, cv::Size(MODEL_IN_H, MODEL_IN_W));
    /* return result */
    return result;
}

/*****************************************
 * Function Name : lane_segmentation
 * Description   : Function to perform over all segmentation
 * Arguments     : Mat frame
 * Return value  : Mat frame
 ******************************************/
cv::Mat lane_segmentation(cv::Mat frame)
{
    float *output;
    int64_t output_size_unet;
    cv::Mat input_frame,output_frame;
    input_frame = frame;

    /* Preprocess time start */
    auto t0 = std::chrono::high_resolution_clock::now();
    cv::Size size(MODEL_IN_H, MODEL_IN_W);
    /*resize the image to the model input size*/
    cv::resize(frame, frame, size);
    /* start pre-processing */
    vector<Mat> rgb_images;
    split(frame, rgb_images);
    Mat m_flat_r = rgb_images[0].reshape(1, 1);
    Mat m_flat_g = rgb_images[1].reshape(1, 1);
    Mat m_flat_b = rgb_images[2].reshape(1, 1);
    Mat matArray[] = {m_flat_r, m_flat_g, m_flat_b};
    Mat frameCHW;
    hconcat(matArray, 3, frameCHW);

    frameCHW.convertTo(frameCHW, CV_32FC3,1.0 / 255.0, 0);
    /*deep copy, if not continuous*/
    if (!frameCHW.isContinuous())
    frameCHW = frameCHW.clone();

    /* Preprocess time ends*/
    auto t1 = std::chrono::high_resolution_clock::now();

    /*start inference using drp runtime*/
    /*Set Pre-processing output to be inference input. */
    runtime.SetInput(0, frameCHW.ptr<float>());
    /* Inference time start */
    auto t2 = std::chrono::high_resolution_clock::now();
    runtime.Run(drpai_freq);
    /* Inference time end */
    auto t3 = std::chrono::high_resolution_clock::now();
    auto inf_duration = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    /* Postprocess time start */
    auto t4 = std::chrono::high_resolution_clock::now();
   
    /* get output buffer */
    auto output_buffer = runtime.GetOutput(0);
    output_size_unet = std::get<2>(output_buffer);
    floatarr.resize(output_size_unet);
     /* Post-processing for FP16 */
    if (InOutDataType::FLOAT16 == std::get<0>(output_buffer))
    {
        /* Extract data in FP16 <uint16_t>. */
        uint16_t *data_ptr = reinterpret_cast<uint16_t *>(std::get<1>(output_buffer));
        for (int n = 0; n < output_size_unet; n++)
        {
            /* Cast FP16 output data to FP32. */
            floatarr[n] = float16_to_float32(data_ptr[n]);
        }
    }
    
    /* convert float32 format to opencv mat image format */ 
    cv::Mat img_mask(MODEL_IN_H,MODEL_IN_W,CV_32F,floatarr.data());
    /* setting minimum threshold to heatmap */ 
    cv::threshold(img_mask,img_mask,TH_PROB,0.0,cv::THRESH_TOZERO);
    cv::normalize(img_mask, img_mask, 0.0, 1.0, cv::NORM_MINMAX);
    /* Scale the float values to 0-255 range for visualization */
    cv::Mat heatmap_scaled;
    img_mask.convertTo(heatmap_scaled, CV_8U, 255.0);
    /* Create a grayscale heatmap */
    cv::applyColorMap(heatmap_scaled, img_mask, cv::COLORMAP_INFERNO);
    cv::cvtColor(img_mask, output_frame, cv::COLOR_RGB2BGR);

    cv::Mat green_image = cv::Mat::zeros(heatmap_scaled.size(), CV_8UC3);
    green_image.setTo(cv::Scalar(0, 255, 0), heatmap_scaled > 0);
    output_frame = green_image;

    /* convert white colour from output frame to green colour */
    output_frame = colour_convert(output_frame);

    cv::resize(input_frame, input_frame, cv::Size(DISP_INF_WIDTH, DISP_INF_HEIGHT));
    cv ::cvtColor(input_frame, input_frame, cv::COLOR_RGB2BGR);
    cv::resize(output_frame, output_frame, cv::Size(DISP_INF_WIDTH, DISP_INF_HEIGHT));
    cv::threshold(output_frame, output_frame, 0.7, 255, 3);

    /* blending both input and ouput frames that have same size and format and combined one single frame */
    cv::addWeighted(input_frame, 1.0, output_frame, 0.5, 0.0, output_frame);

    /* Postprocess time end */
    auto t5 = std::chrono::high_resolution_clock::now();

    auto r_post_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
    auto pre_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    /* resize the output image with respect to output window size */
    cv::cvtColor(output_frame, output_frame, cv::COLOR_RGB2BGR);

    mtx1.lock();
    POST_PROC_TIME = r_post_proc_time/1000.0;
    PRE_PROC_TIME = pre_proc_time/1000.0;
    INF_TIME = inf_duration/1000.0;
    float total_time = float(inf_duration/1000.0) + float(POST_PROC_TIME) + float(pre_proc_time/1000.0);
    TOTAL_TIME = total_time;
    mtx1.unlock();


    return output_frame;
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
 *                      
 * Return value  : media_port, media port that device is connected. 
 ******************************************/
std::string query_device_status(std::string device_type)
{
    std::string media_port = "";
    /* Linux command to be executed */
    const char *command = "v4l2-ctl --list-devices";
    /* Open a pipe to the command and execute it */
    FILE *pipe = popen(command, "r");
    if (!pipe)
    {
        std::cerr << "[ERROR] Unable to open the pipe." << std::endl;
        return media_port;
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
            media_port = std::string(buffer);
            pclose(pipe);
            /* return media port*/
            return media_port;
        }
    }
    pclose(pipe);
    /* return media port*/
    return media_port;
}


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
    int8_t ret = 0;

    printf("Capture Thread Starting\n");

    /* Capture stream of frames from camera using Gstreamer pipeline */
    cap.open(gstreamer_pipeline, CAP_GSTREAMER);
    if (!cap.isOpened())
    {
        std::cerr << "[ERROR] Error opening video stream or camera !" << std::endl;
        return 0;
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
            /*Checks if image frame from Capture Thread is ready.*/
            if (capture_start.load())
            {
                break;
            }
            usleep(WAIT_TIME);
        }

        /* Capture USB camera image and stop updating the capture buffer */
        cap >> g_frame;
        if (g_frame.empty())
        {
            std::cout << "[INFO] Video ended or corrupted frame !\n";
            return 0;
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
    int32_t ret = 0;
    int32_t memcpy_sem_check = 0;

    static int8_t memcpy_flag = 1;
    
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
                memcpy(inf_frame.data, g_frame.data, IMAGE_WIDTH * IMAGE_HEIGHT * BGR_CHANNEL);
                inference_start.store(1); /* Flag for AI Inference Thread. */
                img_processing_start.store(1);
                memcpy_flag = 0;
            }

            if (!img_processing_start.load() && !inference_start.load() && memcpy_flag == 0)
            {
                /* Copy captured image to inference buffer. This will be used in AI Inference Thread. */
                memcpy(img_frame.data, inf_frame.data, IMAGE_WIDTH * IMAGE_HEIGHT * BGR_CHANNEL);
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
    cv::Mat result_image;

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

        result_image = lane_segmentation(inf_frame);
        if (result_image.empty())
        {
            std::cerr << "[ERROR] Inference Not working !!!" << std::endl;
        }
        else
        {
            mtx.lock();
            // Store the result image in the global variable
            g_result_image = result_image.clone();
            mtx.unlock();
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

int8_t R_Main_Process() {
    /* Main Process Variables */
    int8_t main_ret = 0;
    int32_t main_sem_check = 0;
    int8_t ret = 0;
    std::stringstream stream;
    std::string str;
    int32_t baseline = 10;
    cv::Mat output_image = cv::Mat(DISP_OUTPUT_HEIGHT,DISP_OUTPUT_WIDTH , CV_8UC3);
    cv::Mat bgra_image = cv::Mat(DISP_OUTPUT_HEIGHT,DISP_OUTPUT_WIDTH,CV_8UC4);
    cv::Mat pre_image = cv::Mat(DISP_INF_HEIGHT, DISP_INF_WIDTH , CV_8UC3);
    printf("Main Loop Starts\n");

    while(1)
    {
        while (1) 
        {
            /* Get the termination request semaphore value */
            errno = 0;
            ret = sem_getvalue(&terminate_req_sem, &main_sem_check);
            if (0 != ret) {
                fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
                goto err;
            }
            if (1 != main_sem_check) {
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
        pre_image.setTo(cv::Scalar(0, 0, 0));

        if (!g_result_image.empty())
        {
            cv::resize(g_result_image, pre_image, Size(DISP_INF_WIDTH, DISP_INF_HEIGHT));
        }
        else
        {
            cv::resize(img_frame, pre_image, Size(DISP_INF_WIDTH, DISP_INF_HEIGHT));
        }

        pre_image.copyTo(output_image(Rect(0, 60, DISP_INF_WIDTH, DISP_INF_HEIGHT)));

        /* Display TSU value */           
        FILE *fp;
        char buff[16]="";
        float tmp;
        fp = fopen("/sys/class/thermal/thermal_zone1/temp", "r");
        fgets(buff, 16, fp);
        fclose(fp);
        tmp  = (float)atoi(buff)/1000;

        stream.str("");
        stream << "Total Time: " << fixed << setprecision(1) << TOTAL_TIME<<" ms";
        str = stream.str();
        Size tot_time_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_LARGE, LANE_CHAR_THICKNESS, &baseline);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - tot_time_size.width - RIGHT_ALIGN_OFFSET), (T_TIME_STR_Y + tot_time_size.height)), FONT_HERSHEY_SIMPLEX, 
                    CHAR_SCALE_LARGE, Scalar(0, 0, 0), 1.5*LANE_CHAR_THICKNESS);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - tot_time_size.width - RIGHT_ALIGN_OFFSET), (T_TIME_STR_Y + tot_time_size.height)), FONT_HERSHEY_SIMPLEX, 
                    CHAR_SCALE_LARGE, Scalar(0, 255, 0), LANE_CHAR_THICKNESS);
        stream.str("");
        stream << "Pre-Proc: " << fixed << setprecision(1)<< PRE_PROC_TIME<<" ms";
        str = stream.str();
        Size pre_proc_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, LANE_CHAR_THICKNESS, &baseline);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - pre_proc_size.width - RIGHT_ALIGN_OFFSET), (PRE_TIME_STR_Y + pre_proc_size.height)), FONT_HERSHEY_SIMPLEX, 
                    CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*LANE_CHAR_THICKNESS);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - pre_proc_size.width - RIGHT_ALIGN_OFFSET), (PRE_TIME_STR_Y + pre_proc_size.height)), FONT_HERSHEY_SIMPLEX, 
                    CHAR_SCALE_SMALL, Scalar(255, 255, 255), LANE_CHAR_THICKNESS);
        stream.str("");
        stream << "Inference: "<< fixed << setprecision(1) << INF_TIME<<" ms";
        str = stream.str();
        Size inf_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, LANE_CHAR_THICKNESS, &baseline);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - inf_size.width - RIGHT_ALIGN_OFFSET), (I_TIME_STR_Y + inf_size.height)), FONT_HERSHEY_SIMPLEX, 
                    CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*LANE_CHAR_THICKNESS);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - inf_size.width - RIGHT_ALIGN_OFFSET), (I_TIME_STR_Y + inf_size.height)), FONT_HERSHEY_SIMPLEX, 
                    CHAR_SCALE_SMALL, Scalar(255, 255, 255), LANE_CHAR_THICKNESS);
        stream.str("");
        stream << "Post-Proc: "<< fixed << setprecision(1) << POST_PROC_TIME<<" ms";
        str = stream.str();
        Size post_proc_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, LANE_CHAR_THICKNESS, &baseline);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - post_proc_size.width - RIGHT_ALIGN_OFFSET), (P_TIME_STR_Y + post_proc_size.height)), FONT_HERSHEY_SIMPLEX, 
                    CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*LANE_CHAR_THICKNESS);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - post_proc_size.width - RIGHT_ALIGN_OFFSET), (P_TIME_STR_Y + post_proc_size.height)), FONT_HERSHEY_SIMPLEX, 
                    CHAR_SCALE_SMALL, Scalar(255, 255, 255), LANE_CHAR_THICKNESS);
        stream.str("");         
        stream << "Temperature: "<< fixed <<setprecision(1) << tmp << "C";         
        str = stream.str();         
        Size temp_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, LANE_CHAR_THICKNESS, &baseline);         
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - temp_size.width - RIGHT_ALIGN_OFFSET), (TEMP_STR_Y + temp_size.height)), FONT_HERSHEY_SIMPLEX,
                    CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*LANE_CHAR_THICKNESS);        
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - temp_size.width - RIGHT_ALIGN_OFFSET), (TEMP_STR_Y + temp_size.height)), FONT_HERSHEY_SIMPLEX, 
                    CHAR_SCALE_SMALL, Scalar(255, 255, 255), LANE_CHAR_THICKNESS);
        cv::cvtColor(output_image, bgra_image, cv::COLOR_BGR2BGRA);
        wayland.commit(bgra_image.data, NULL);

        img_processing_start.store(0);
        
    
    }

err:
    sem_trywait(&terminate_req_sem);
    main_ret = 1;
    goto main_proc_end;
/*Main Processing Termination*/
main_proc_end:
    printf("Main Process Terminated\n");
    return main_ret;
}

int main(int argc, char **argv)
{

    int32_t create_thread_ai = -1;
    int32_t create_thread_key = -1;
    int32_t create_thread_capture = -1;
    int32_t create_thread_memcpy  = -1;
    int8_t ret_main = 0;
    int32_t ret = 0;
    int8_t main_proc = 0;
    int32_t sem_create = -1;
    int drpai_fd;
    InOutDataType input_data_type;
    std::string input_source = argv[1];
    std::cout << "Starting Road Lane Segmentation Application" << std::endl;
    
    /*Disable OpenCV Accelerator due to the use of multithreading */
    unsigned long OCA_list[16];
    for (int i=0; i < 16; i++)
    {
        OCA_list[i] = 0;
    }
    OCA_Activate( &OCA_list[0] );

    if (strcmp(argv[1],"USB")==0)
        {   
            if (argc >= 3 )
            {
                drpai_freq = atoi(argv[2]);
                if ((1 <= drpai_freq) && (127 >= drpai_freq))
                {
                    printf("Argument : <AI-MAC_freq_factor> = %d\n", drpai_freq);
                }
                else
                {
                    fprintf(stderr,"[ERROR] Invalid Command Line Argument : <AI-MAC_freq_factor>=%d\n", drpai_freq);
                    goto end_main;
                }

            }
            else
            {
                drpai_freq = DRPAI_FREQ;
            }
        }
        else
        {
            std::cout<<"Support for USB mode only."<<std::endl;
            goto end_main;
        }

        if (argc>3)
        {
            std::cerr << "[ERROR] Wrong number Arguments are passed " << std::endl;
            return 1;
        }


    errno = 0;
    drpai_fd = open("/dev/drpai0", O_RDWR);
    if (0 > drpai_fd)
    {
        std::cerr << "[ERROR] Failed to open DRP-AI Driver : errno=" << errno << std::endl;
        ret_main = -1;
        goto end_main;
    }

    /* Initialize DRP-AI (Get DRP-AI memory address and set DRP-AI frequency)*/
    drpaimem_addr_start = init_drpai(drpai_fd);
    if (drpaimem_addr_start == 0) 
    {
        fprintf(stderr, "[ERROR] Failed to get DRP-AI memory area start address.\n");
        close(drpai_fd);
	    goto end_close_drpai;
    }
    runtime_status = runtime.LoadModel(model_dir, drpaimem_addr_start+DRPAI_MEM_OFFSET);
    if(!runtime_status)
    {
        fprintf(stderr, "[ERROR] Failed to load model.\n");
	    close(drpai_fd);
        goto end_close_drpai;
    }
    std::cout << "[INFO] loaded runtime model :" << model_dir << "\n\n";

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

    /* Initialize wayland */
    ret = wayland.init(DISP_OUTPUT_WIDTH, DISP_OUTPUT_HEIGHT, BGRA_CHANNEL);
    if(0 != ret)
    {
        fprintf(stderr, "[ERROR] Failed to initialize Image for Wayland\n");
        goto end_close_drpai;
    }
    /* Get input Source WS/VIDEO/CAMERA */
    switch (input_source_map[input_source])
    {
        /* Input Source : USB Camera */
        case 1:
        {
            std::cout << "[INFO] USB CAMERA \n";
            /* check the status of device */
            std::string media_port = query_device_status("usb");
            /* g-streamer pipeline to read input image source */
            gstreamer_pipeline = "v4l2src device=" + media_port + " ! video/x-raw, width=640, height=480 ! videoconvert ! appsink";
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
            break;
        }

    }
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

    if (0 == sem_create)
    {
        sem_destroy(&terminate_req_sem);
    }

    /* Exit the program */
    wayland.exit();
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
