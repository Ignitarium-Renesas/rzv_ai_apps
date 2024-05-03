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
* File Name    : driver_monitoring_system.cpp
* Version      : 1.1.0
* Description  : DRP-AI TVM[*1] Application Example
***********************************************************************************************************************/

/*****************************************
* includes
******************************************/
#include "define.h"
#include "box.h"
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
MeraDrpRuntimeWrapper runtime1;

/*Global Variables*/
static float drpai_output_buf[num_inf_out];
static float drpai_output_buf1[INF_OUT_SIZE];

static Wayland wayland;
static pthread_t ai_inf_thread;
static pthread_t kbhit_thread;
static sem_t terminate_req_sem;
static int32_t drpai_freq;


static atomic<uint8_t> hdmi_obj_ready   (0);

static uint32_t disp_time = 0;

std::string media_port;
std::string gstreamer_pipeline;

std::vector<float> floatarr(1);

uint64_t drpaimem_addr_start = 0;
bool runtime_status = false; 
bool runtime_status1 = false;
static vector<detection> det;

float fps = 0;
float TOTAL_TIME = 0;
float TOTAL_TIME_DEEPPOSE = 0;
int32_t HEAD_COUNT= 0;
int fd;

float POST_PROC_TIME_TINYYOLO =0;
float POST_PROC_TIME_DEEPPOSE =0;
float PRE_PROC_TIME_TINYYOLO =0;
float PRE_PROC_TIME_DEEPPOSE =0;
float INF_TIME_DEEPPOSE = 0;
float INF_TIME_TINYYOLO = 0;

uint8_t yawn_flag=0;
uint8_t blink_flag=0;

// Cropping parameters
static int16_t cropx1[NUM_MAX_FACE];
static int16_t cropy1[NUM_MAX_FACE];
static int16_t cropx2[NUM_MAX_FACE];
static int16_t cropy2[NUM_MAX_FACE];
static int16_t cropx[NUM_MAX_FACE];
static int16_t cropy[NUM_MAX_FACE];
static int16_t cropw[NUM_MAX_FACE];
static int16_t croph[NUM_MAX_FACE];

static uint32_t deeppose_preds[NUM_OUTPUT_KEYPOINT][2];
static uint16_t id_x[NUM_OUTPUT_KEYPOINT][NUM_MAX_FACE];
static uint16_t id_y[NUM_OUTPUT_KEYPOINT][NUM_MAX_FACE];
static uint16_t id_x_local[NUM_OUTPUT_KEYPOINT][NUM_MAX_FACE]; 
static uint16_t id_y_local[NUM_OUTPUT_KEYPOINT][NUM_MAX_FACE];

/*ML model inferencing*/
static cv::Ptr<cv::ml::RTrees> tree = cv::ml::RTrees::create();
static cv::Ptr<cv::ml::RTrees> dtree = tree->load(xml);
static std::string random_forest_preds[NUM_MAX_FACE];


/*Global frame */
Mat g_frame;
VideoCapture cap;

cv::Mat output_image;

/* Map to store input source list */
std::map<std::string, int> input_source_map ={    
    {"USB", 1}
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
 * Function Name     : load_label_file
 * Description       : Load label list text file and return the label list that contains the label.
 * Arguments         : label_file_name = filename of label list. must be in txt format
 * Return value      : vector<string> list = list contains labels
 *                     empty if error occurred
 ******************************************/
vector<string> load_label_file(string label_file_name)
{
    vector<string> list = {};
    vector<string> empty = {};
    ifstream infile(label_file_name);

    if (!infile.is_open())
    {
        return list;
    }

    string line = "";
    while (getline(infile, line))
    {
        list.push_back(line);
        if (infile.fail())
        {
            return empty;
        }
    }

    return list;
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

    for (i = 0 ; i < n; i++)
    {
        prev_layer_num += NUM_BB *(NUM_CLASS + 5)* num_grids[i] * num_grids[i];
    }
    return prev_layer_num + b *(NUM_CLASS + 5)* num * num + y * num + x;
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
* Function Name : R_Inf_RandomForest
* Description   : CPU inference for Random Forest
* Arguments     : floatarr = drpai output address
*                 n_pers = number of the face detected
* Return value  : -
******************************************/
static void R_Inf_RandomForest(float* floatarr, uint8_t n_pers)
{
    cv::Mat X = cv::Mat(1, INF_OUT_SIZE, CV_32F, floatarr);
    cv::Mat preds = {};
    dtree->predict(X, preds);
    random_forest_preds[n_pers] = label_list1[((int)(preds.at<float>(0)) - 1)];
    return;
}


/*****************************************
* Function Name : R_Post_Proc_DeepPose
* Description   : CPU post-processing for DeepPose
* Arguments     : floatarr = drpai output address
*                 n_pers = number of the face detected
* Return value  : -
******************************************/
static void R_Post_Proc_DeepPose(float* floatarr, uint8_t n_pers)
{
    float scale[] = {(float)(cropw[n_pers]), (float)(croph[n_pers])};
    for (int i = 0; i < NUM_OUTPUT_KEYPOINT; i++)
    {
        deeppose_preds[i][0] = (int32_t)(floatarr[2*i] * scale[0]);
        deeppose_preds[i][1] = (int32_t)(floatarr[2*i+1] * scale[1]);
    }
    return;
}

/*****************************************
* Function Name : R_DeepPose_Coord_Convert
* Description   : Convert the post processing result into drawable coordinates
* Arguments     : n_pers = number of the detected face
* Return value  : -
******************************************/
static void R_DeepPose_Coord_Convert(uint8_t n_pers)
{
    /* Render skeleton on image and print their details */
    int32_t posx;
    int32_t posy;
    int8_t i;
    

    for (i = 0; i < NUM_OUTPUT_KEYPOINT; i++)
    {
       

        posx = (int32_t)(deeppose_preds[i][0] + 0.5) + cropx[n_pers] + OUTPUT_ADJ_X;
        posy = (int32_t)(deeppose_preds[i][1] + 0.5) + cropy[n_pers] + OUTPUT_ADJ_Y;
        
        /* Make sure the coordinates are not off the screen. */
        posx = (posx < 0) ? 0 : posx;
        posx = (posx > IMREAD_IMAGE_WIDTH - KEY_POINT_SIZE -1 ) ? IMREAD_IMAGE_WIDTH -KEY_POINT_SIZE -1 : posx;
        posy = (posy < 0) ? 0 : posy;
        posy = (posy > IMREAD_IMAGE_HEIGHT -KEY_POINT_SIZE -1) ? IMREAD_IMAGE_HEIGHT -KEY_POINT_SIZE -1 : posy;
        id_x_local[i][n_pers] = posx;
        id_y_local[i][n_pers] = posy;
    }
    return;
}

static double calculate_euclidean_distance(int x1, int x2, int y1, int y2)
{
    double x = x1 - x2; // calculating number to square in next step
    double y = y1 - y2;
    double euclidean_distance;

    euclidean_distance = pow(x, 2) + pow(y, 2); // calculating Euclidean distance
    euclidean_distance = sqrt(euclidean_distance);
    return euclidean_distance;
}


/*****************************************
 * Function Name : R_Post_Proc
 * Description   : Process CPU post-processing for YOLOv3
 * Arguments     : floatarr = drpai output address
 * Return value  : -
 ******************************************/
void R_Post_Proc(float* floatarr)
{
    int32_t result_cnt =0;
    /* Following variables are required for correct_yolo/region_boxes in Darknet implementation*/
    /* Note: This implementation refers to the "darknet detector test" */
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
    /* Clear the detected result list */
    det.clear();

    for (n = 0; n<NUM_INF_OUT_LAYER; n++)
    {
        num_grid = num_grids[n];
        anchor_offset = 2 * NUM_BB * (NUM_INF_OUT_LAYER - (n + 1));

        for (b = 0;b<NUM_BB;b++)
        {
            for (y = 0;y<num_grid;y++)
            {
                for (x = 0;x<num_grid;x++)
                {
                    offs = yolo_offset(n, b, y, x);
                    tx = floatarr[offs];
                    ty = floatarr[yolo_index(n, offs, 1)];
                    tw = floatarr[yolo_index(n, offs, 2)];
                    th = floatarr[yolo_index(n, offs, 3)];
                    tc = floatarr[yolo_index(n, offs, 4)];

                    /* Compute the bounding box */
                    /*get_yolo_box/get_region_box in paper implementation*/
                    center_x = ((float) x + sigmoid(tx)) / (float) num_grid;
                    center_y = ((float) y + sigmoid(ty)) / (float) num_grid;
                    box_w = (float) exp(tw) * anchors[anchor_offset+2*b+0] / (float) MODEL_IN_W;
                    box_h = (float) exp(th) * anchors[anchor_offset+2*b+1] / (float) MODEL_IN_W;

                    /* Adjustment for VGA size */
                    /* correct_yolo/region_boxes */
                    center_x = (center_x - (MODEL_IN_W - new_w) / 2. / MODEL_IN_W) / ((float) new_w / MODEL_IN_W);
                    center_y = (center_y - (MODEL_IN_H - new_h) / 2. / MODEL_IN_H) / ((float) new_h / MODEL_IN_H);
                    box_w *= (float) (MODEL_IN_W / new_w);
                    box_h *= (float) (MODEL_IN_H / new_h);

                    center_x = round(center_x * DRPAI_IN_WIDTH);
                    center_y = round(center_y * DRPAI_IN_HEIGHT);
                    box_w = round(box_w * DRPAI_IN_WIDTH);
                    box_h = round(box_h * DRPAI_IN_HEIGHT);

                    objectness = sigmoid(tc);

                    Box bb = {center_x, center_y, box_w, box_h};
                    /* Get the class prediction */
                    for (i = 0;i < NUM_CLASS;i++)
                    {
                        classes[i] = sigmoid(floatarr[yolo_index(n, offs, 5+i)]);
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
                    /*printf("probability : %f", probability);*/
                    if (probability >= TH_PROB)
                    {
                        d = {bb, pred_class, probability};
                        det.push_back(d);
                    }
                }
            }
        }
    }
    /* Non-Maximum Supression filter */
    filter_boxes_nms(det, det.size(), TH_NMS);
    for (i = 0; i < det.size(); i++)
    {
        /* Skip the overlapped bounding boxes */
        if (det[i].prob == 0)
        {
            continue;
        }
        else{
            result_cnt++;
            if(result_cnt > 2)
            {
                break;
            }
        }
    }
    HEAD_COUNT = result_cnt;

    return ;
}

/*****************************************
 * Function Name : DMS
 * Description   : Function to perform over all detection
 * Arguments     : -
 * Return value  : 0 if succeeded
 *               not 0 otherwise
 ******************************************/
int DMS()
{   
    int wait_key;
     /* Temp frame */
    Mat frame1;

    Size size(MODEL_IN_H, MODEL_IN_W);
    auto t0 = std::chrono::high_resolution_clock::now();
    /*resize the image to the model input size*/
    resize(g_frame, frame1, size);

    /* changing channel from hwc to chw */
    vector<Mat> rgb_images;
    split(frame1, rgb_images);
    Mat m_flat_r = rgb_images[0].reshape(1, 1);
    Mat m_flat_g = rgb_images[1].reshape(1, 1);
    Mat m_flat_b = rgb_images[2].reshape(1, 1);
    Mat matArray[] = {m_flat_r, m_flat_g, m_flat_b};
    Mat frameCHW;
    hconcat(matArray, 3, frameCHW);
    /*convert to FP32*/
    frameCHW.convertTo(frameCHW, CV_32FC3);

    /* normailising  pixels */
    divide(frameCHW, 255.0, frameCHW);

    /* DRP AI input image should be continuous buffer */
    if (!frameCHW.isContinuous())
        frameCHW = frameCHW.clone();

    Mat frame = frameCHW;
    int ret = 0;

    /* Preprocess time ends for tinyyolo model*/
    auto t1 = std::chrono::high_resolution_clock::now();
    
     /* tinyyolov2 inference*/
    /*start inference using drp runtime*/
    runtime.SetInput(0, frame.ptr<float>());
    
    /* Inference start time for tinyyolo model*/
    auto t2 = std::chrono::high_resolution_clock::now();
    runtime.Run(drpai_freq);
    /* Inference time end for tinyyolo model */
    auto t3 = std::chrono::high_resolution_clock::now();
    auto inf_duration = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    /* Postprocess time start for tinyyolo model */
    auto t4 = std::chrono::high_resolution_clock::now();
    /*load inference out on drpai_out_buffer*/
    int32_t i = 0;
    int32_t output_num = 0;
    std::tuple<InOutDataType, void *, int64_t> output_buffer;
    int64_t output_size;
    uint32_t size_count = 0;

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

     /* Do post process to get bounding boxes */
    R_Post_Proc(drpai_output_buf);
    /*/* Postprocess time end for tinyyolo model*/
    auto t5 = std::chrono::high_resolution_clock::now();
    if (HEAD_COUNT > 0)
    {
       float POST_PROC_TIME_DEEPPOSE_MICRO =0;
       float PRE_PROC_TIME_DEEPPOSE_MICRO =0;
       float INF_TIME_DEEPPOSE_MICRO = 0;
        
     for (int i = 0 ; i < HEAD_COUNT; i++)
     {
        /* Preprocess time start for deeppose model*/
        auto t0_deeppose = std::chrono::high_resolution_clock::now();
       
        cropx1[i] = (int)det[i].bbox.x - round((int)det[i].bbox.w/2 );
        cropy1[i] = (int)det[i].bbox.y - round((int)det[i].bbox.h/2 );
        cropx2[i] = (int)det[i].bbox.x + round((int)det[i].bbox.w/2 ) - 1;
        cropy2[i] = (int)det[i].bbox.y + round((int)det[i].bbox.h/2 ) - 1;

        /* Check the bounding box is in the image range */
        cropx1[i] = cropx1[i] < 1 ? 1 : cropx1[i];
        cropx2[i] = ((DRPAI_IN_WIDTH - 2) < cropx2[i]) ? (DRPAI_IN_WIDTH- 2) : cropx2[i];
        cropy1[i] = cropy1[i] < 1 ? 1 : cropy1[i];
        cropy2[i] = ((DRPAI_IN_HEIGHT - 2) < cropy2[i]) ? (DRPAI_IN_HEIGHT - 2) : cropy2[i];

        Mat cropped_image = g_frame(Range(cropy1[i],cropy2[i]), Range(cropx1[i],cropx2[i]));
        
        Mat frame1_deeppose;

        Size size(256, 256);

        /*resize the image to the model input size*/
        resize(cropped_image, frame1_deeppose, size);
        vector<Mat> rgb_images_deeppose;
        split(frame1_deeppose, rgb_images_deeppose);
        Mat m_flat_r_deeppose = rgb_images_deeppose[0].reshape(1, 1);
        Mat m_flat_g_deeppose = rgb_images_deeppose[1].reshape(1, 1);
        Mat m_flat_b_deeppose = rgb_images_deeppose[2].reshape(1, 1);
        Mat matArray_deeppose[] = {m_flat_r_deeppose, m_flat_g_deeppose, m_flat_b_deeppose};
        Mat frameCHW_deeppose;
        hconcat(matArray_deeppose, 3, frameCHW_deeppose);
        /*convert to FP32*/
        frameCHW_deeppose.convertTo(frameCHW_deeppose, CV_32FC3);

        /* normailising  pixels */
        divide(frameCHW_deeppose, 255.0, frameCHW_deeppose);

        /* DRP AI input image should be continuous buffer */
        if (!frameCHW_deeppose.isContinuous())
            frameCHW_deeppose = frameCHW_deeppose.clone();

        Mat frame_deeppose = frameCHW_deeppose;
        auto t1_deeppose = std::chrono::high_resolution_clock::now();

            /* deeppose inference*/
            runtime1.SetInput(0, frame_deeppose.ptr<float>());
            /*inference start time for deeppose model*/
            auto t2_deeppose = std::chrono::high_resolution_clock::now();
            runtime1.Run(drpai_freq);
            /*inference end time for deeppose model*/
            auto t3_deeppose = std::chrono::high_resolution_clock::now();
            auto inf_duration_deeppose = std::chrono::duration_cast<std::chrono::microseconds>(t3_deeppose - t2_deeppose).count();
        
            /*Postprocess time start for deeppose model*/
            auto t4_deeppose = std::chrono::high_resolution_clock::now();
            /*load inference out on drpai_out_buffer*/
            int32_t l = 0;
            int32_t output_num_deeppose = 0;
            std::tuple<InOutDataType, void *, int64_t> output_buffer_deeppose;
            int64_t output_size_deeppose;
            uint32_t size_count_deeppose = 0;

            /* Get the number of output of the target model. */
            output_num_deeppose = runtime1.GetNumOutput();
            size_count_deeppose = 0;
            /*GetOutput loop*/
            for (l = 0; l < output_num_deeppose; l++)
            {
                /* output_buffer below is tuple, which is { data type, address of output data, number of elements } */
                output_buffer_deeppose = runtime1.GetOutput(l);
                /*Output Data Size = std::get<2>(output_buffer). */
                output_size_deeppose = std::get<2>(output_buffer_deeppose);

                /*Output Data Type = std::get<0>(output_buffer)*/
                if (InOutDataType::FLOAT16 == std::get<0>(output_buffer_deeppose))
                {
                    /*Output Data = std::get<1>(output_buffer)*/
                    uint16_t *data_ptr_deeppose = reinterpret_cast<uint16_t *>(std::get<1>(output_buffer_deeppose));
                    for (int j = 0; j < output_size_deeppose; j++)
                    {
                        /*FP16 to FP32 conversion*/
                        drpai_output_buf1[j + size_count_deeppose] = float16_to_float32(data_ptr_deeppose[j]);
                    }
                }
                else if (InOutDataType::FLOAT32 == std::get<0>(output_buffer_deeppose))
                {
                    /*Output Data = std::get<1>(output_buffer)*/
                    float *data_ptr_deeppose = reinterpret_cast<float *>(std::get<1>(output_buffer_deeppose));
                    for (int j = 0; j < output_size_deeppose; j++)
                    {
                        drpai_output_buf1[j + size_count_deeppose] = data_ptr_deeppose[j];
                    }
                }
                else
                {
                    std::cerr << "[ERROR] Output data type : not floating point." << std::endl;
                    ret = -1;
                    break;
                }
                size_count_deeppose += output_size_deeppose;
            }


        if (ret != 0)
        {
            std::cerr << "[ERROR] DRP Inference Not working !!! " << std::endl;
            return -1;
        }
        croph[i] = det[i].bbox.h   + CROP_ADJ_X ;
        cropw[i] = det[i].bbox.w   + CROP_ADJ_Y ;
        /*Checks that cropping height and width does not exceeds image dimension*/
        if(croph[i] < 1)
        {
            croph[i] = 1;
        }
        else if(croph[i] > IMREAD_IMAGE_HEIGHT)
        {
        croph[i] = IMREAD_IMAGE_HEIGHT;
        }
        else
        {
         /*Do Nothing*/
        }
        if(cropw[i] < 1)
        {
        cropw[i] = 1;
        }
        else if(cropw[i] > IMREAD_IMAGE_WIDTH)
        {
        cropw[i] = IMREAD_IMAGE_WIDTH;
        }
        else
        {
        /*Do Nothing*/
        }
        /*Compute Cropping Y Position based on Detection Result*/
        /*If Negative Cropping Position*/
        if(det[i].bbox.y < (croph[i]/2))
        {
         cropy[i] = 0;
        }
        else if(det[i].bbox.y > (IMREAD_IMAGE_HEIGHT-croph[i]/2)) /*If Exceeds Image Area*/
        {
        cropy[i] = IMREAD_IMAGE_HEIGHT-croph[i];
        }
        else
        {
        cropy[i] = (int16_t)det[i].bbox.y - croph[i]/2;
        }
        /*Compute Cropping X Position based on Detection Result*/
        /*If Negative Cropping Position*/
        if(det[i].bbox.x < (cropw[i]/2))
        {
            cropx[i] = 0;
        }
        else if(det[i].bbox.x > (IMREAD_IMAGE_WIDTH-cropw[i]/2)) /*If Exceeds Image Area*/
        {
            cropx[i] = IMREAD_IMAGE_WIDTH-cropw[i];
        }
        else
        {
            cropx[i] = (int16_t)det[i].bbox.x - cropw[i]/2;
        }
        /*Checks that combined cropping position with width and height does not exceed the image dimension*/
        if(cropx[i] + cropw[i] > IMREAD_IMAGE_WIDTH)
        {
            cropw[i] = IMREAD_IMAGE_WIDTH - cropx[i];
        }
        if(cropy[i] + croph[i] > IMREAD_IMAGE_HEIGHT)
        {
            croph[i] = IMREAD_IMAGE_HEIGHT - cropy[i];
        }

        R_Post_Proc_DeepPose(drpai_output_buf1,i);
        R_DeepPose_Coord_Convert(i);
        R_Inf_RandomForest(drpai_output_buf1,i);
        /*Postprocess time end for deeppose model*/
            auto t5_deeppose = std::chrono::high_resolution_clock::now();

            auto r_post_proc_time_deeppose = std::chrono::duration_cast<std::chrono::microseconds>(t5_deeppose - t4_deeppose).count();
            auto pre_proc_time_deeppose = std::chrono::duration_cast<std::chrono::microseconds>(t1_deeppose - t0_deeppose).count();
        
            POST_PROC_TIME_DEEPPOSE_MICRO = POST_PROC_TIME_DEEPPOSE_MICRO + r_post_proc_time_deeppose;
            PRE_PROC_TIME_DEEPPOSE_MICRO = PRE_PROC_TIME_DEEPPOSE_MICRO + pre_proc_time_deeppose;
            INF_TIME_DEEPPOSE_MICRO = INF_TIME_DEEPPOSE_MICRO + inf_duration_deeppose;
        }

        POST_PROC_TIME_DEEPPOSE = POST_PROC_TIME_DEEPPOSE_MICRO/1000.0;
        PRE_PROC_TIME_DEEPPOSE = PRE_PROC_TIME_DEEPPOSE_MICRO/1000.0;
        INF_TIME_DEEPPOSE = INF_TIME_DEEPPOSE_MICRO/1000.0;
        float total_time = float(POST_PROC_TIME_DEEPPOSE_MICRO) + float(PRE_PROC_TIME_DEEPPOSE_MICRO) + float(INF_TIME_DEEPPOSE_MICRO);
        TOTAL_TIME_DEEPPOSE = total_time/1000.0;    
    }
    auto r_post_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
    auto pre_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    POST_PROC_TIME_TINYYOLO = r_post_proc_time/1000.0;
    PRE_PROC_TIME_TINYYOLO = pre_proc_time/1000.0;
    INF_TIME_TINYYOLO = inf_duration/1000.0;

    float total_time_tinyyolo = float(POST_PROC_TIME_TINYYOLO) + float(PRE_PROC_TIME_TINYYOLO) + float(INF_TIME_TINYYOLO);
    TOTAL_TIME = TOTAL_TIME_DEEPPOSE + total_time_tinyyolo;

    /*Calculating the fps*/
    return 0;
}


void click_event(int event, int x, int y, int flags, void* userdata)
{
    if (event == EVENT_LBUTTONDBLCLK)
    {
        std::cout<<"Exiting the event "<<std::endl;
        exit(0);
    }
}

/*****************************************
 * Function Name : capture_frame
 * Description   : function to open camera gstreamer pipeline.
 * Arguments     : string cap_pipeline input pipeline
 ******************************************/
void capture_frame(std::string gstreamer_pipeline )
{
    stringstream stream;
     int32_t inf_sem_check = 0;
    string str = "";
    int32_t baseline = 10;
    int32_t ret = 0;
    int32_t i = 0;
    int32_t x = 0;
    uint8_t * img_buffer0;
    img_buffer0 = (unsigned char*) (malloc(DISP_OUTPUT_WIDTH*DISP_OUTPUT_HEIGHT*BGRA_CHANNEL));
    int wait_key;
    /* Capture stream of frames from camera using Gstreamer pipeline */
    cap.open(gstreamer_pipeline, CAP_GSTREAMER);
    if (!cap.isOpened())
    {
        std::cerr << "[ERROR] Error opening video stream or camera !" << std::endl;
        return;
    }
    while (true)
    {
        cap >> g_frame;
        cv::Mat output_image(DISP_OUTPUT_HEIGHT,DISP_OUTPUT_WIDTH , CV_8UC3, cv::Scalar(0, 0, 0));
        fps = cap.get(CAP_PROP_FPS);
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
        if (g_frame.empty())
        {
            std::cout << "[INFO] Video ended or corrupted frame !\n";
            return;
        }
        else
        {
            int ret = DMS();
            if (ret != 0)
            {
                std::cerr << "[ERROR] Inference Not working !!! " << std::endl;
            }
                memcpy(id_x, id_x_local, sizeof(id_x_local));
        memcpy(id_y, id_y_local,sizeof(id_y_local));

        double lips_width;
        double lips_height;
        double left_eye_vertical_distance;
        double left_eye_horizantal_distance;
        double right_eye_vertical_distance;
        double right_eye_horizantal_distance;
        double ear_right;
        double ear_left;
        if(HEAD_COUNT > 0)
        {
            for(uint8_t i=0; i < HEAD_COUNT; i++)
            {    
                
                if(random_forest_preds[i]== "CENTER") 
                {   

                    lips_width = calculate_euclidean_distance(id_x[76][i], id_x[82][i], id_y[76][i], id_y[82][i]);
                    lips_height = calculate_euclidean_distance(id_x[79][i], id_x[85][i], id_y[79][i], id_y[85][i]);       
                    if(lips_width/lips_height < 2.0)
                    {
                    yawn_flag=2;
                    }
                    else 
                    {
                    yawn_flag=1;
                    }
                    left_eye_vertical_distance = calculate_euclidean_distance(id_x[62][i], id_x[66][i], id_y[62][i], id_y[66][i]);
                    right_eye_vertical_distance = calculate_euclidean_distance(id_x[70][i], id_x[74][i], id_y[70][i], id_y[74][i]);
                    left_eye_horizantal_distance = calculate_euclidean_distance(id_x[60][i], id_x[64][i], id_y[60][i], id_y[64][i]);
                    right_eye_horizantal_distance = calculate_euclidean_distance(id_x[68][i], id_x[72][i], id_y[68][i], id_y[72][i]);

                    ear_right = right_eye_vertical_distance / right_eye_horizantal_distance;
                    ear_left = left_eye_vertical_distance / left_eye_horizantal_distance;

                    if((ear_left < 0.12) && (ear_right < 0.12))
                    {
                    blink_flag=2;

                    }
                    else
                    {
                    blink_flag=1;
                    }
                }
            }
        }
        if( HEAD_COUNT > 0 )
        {
 
            if(yawn_flag==2)
            {
                stream.str("");
                stream << "Yawn Detected"; 
                str = stream.str();
                putText(g_frame, str,Point(YAWN_STR_X , YAWN_STR_Y), FONT_HERSHEY_SIMPLEX, 
                            DMS_CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*DMS_CHAR_THICKNESS);
                putText(g_frame, str,Point(YAWN_STR_X , YAWN_STR_Y), FONT_HERSHEY_SIMPLEX, 
                            DMS_CHAR_SCALE_SMALL, Scalar(0, 255, 255), DMS_CHAR_THICKNESS);
            }      
            if(blink_flag==2)
            {
                stream.str("");
                stream << "Blink Detected";
                str = stream.str();
                putText(g_frame, str,Point(BLINK_STR_X , BLINK_STR_Y), FONT_HERSHEY_SIMPLEX, 
                            DMS_CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*DMS_CHAR_THICKNESS);
                putText(g_frame, str,Point(BLINK_STR_X , BLINK_STR_Y), FONT_HERSHEY_SIMPLEX, 
                            DMS_CHAR_SCALE_SMALL, Scalar(0, 255, 255), DMS_CHAR_THICKNESS);
            }
            for(i=0; i<HEAD_COUNT; i++)
            {
                stream.str("");
                stream << "Head Pose " << i+1 << ": " << random_forest_preds[i];
                str = stream.str();
                putText(g_frame, str,Point(HEAD_POSE_STR_X , HEAD_POSE_STR_Y), FONT_HERSHEY_SIMPLEX,
                            DMS_CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*DMS_CHAR_THICKNESS);
                putText(g_frame, str,Point(HEAD_POSE_STR_X , HEAD_POSE_STR_Y), FONT_HERSHEY_SIMPLEX,
                            DMS_CHAR_SCALE_SMALL, Scalar(0, 255, 255), DMS_CHAR_THICKNESS);

            }
        }
            stream.str("");
            stream << "Camera Frame Rate : "<< fixed << setprecision(1) << fps <<" fps ";
            str = stream.str();
            Size camera_rate_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - camera_rate_size.width - RIGHT_ALIGN_OFFSET), (FPS_STR_Y + camera_rate_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - camera_rate_size.width - RIGHT_ALIGN_OFFSET), (FPS_STR_Y + camera_rate_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);

            stream.str("");
            stream << "Total Time: " << fixed << setprecision(2)<< TOTAL_TIME <<" ms";
            str = stream.str();
            Size tot_time_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_LARGE, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - tot_time_size.width - RIGHT_ALIGN_OFFSET), (T_TIME_STR_Y + tot_time_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_LARGE, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - tot_time_size.width - RIGHT_ALIGN_OFFSET), (T_TIME_STR_Y + tot_time_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_LARGE, Scalar(0, 255, 0), HC_CHAR_THICKNESS);

            stream.str("");
            stream << "Yolov3";
            str = stream.str();
            Size yolov3_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - yolov3_size.width - RIGHT_ALIGN_OFFSET), (MODEL_NAME_1_Y + yolov3_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - yolov3_size.width - RIGHT_ALIGN_OFFSET), (MODEL_NAME_1_Y + yolov3_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);

            stream.str("");
            stream << "Pre-Proc: "  << fixed << setprecision(2)<< PRE_PROC_TIME_TINYYOLO<<" ms";
            str = stream.str();
            Size pre_proc_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - pre_proc_size.width - RIGHT_ALIGN_OFFSET), (PRE_TIME_STR_Y + pre_proc_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - pre_proc_size.width - RIGHT_ALIGN_OFFSET), (PRE_TIME_STR_Y + pre_proc_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);
            stream.str("");
            stream << "Inference: " << fixed << setprecision(2)<< INF_TIME_TINYYOLO<<" ms";
            str = stream.str();
            Size inf_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - inf_size.width - RIGHT_ALIGN_OFFSET), (I_TIME_STR_Y + inf_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - inf_size.width - RIGHT_ALIGN_OFFSET), (I_TIME_STR_Y + inf_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);
            stream.str("");
            stream << "Post-Proc: " << fixed << setprecision(2) << POST_PROC_TIME_TINYYOLO << " ms";
            str = stream.str();
            Size post_proc_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - post_proc_size.width - RIGHT_ALIGN_OFFSET), (P_TIME_STR_Y + post_proc_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - post_proc_size.width - RIGHT_ALIGN_OFFSET), (P_TIME_STR_Y + post_proc_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);
            
            /*Deeppose model Timings*/
            stream.str("");
            stream << "Deeppose";
            str = stream.str();
            Size deeppose_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - deeppose_size.width - RIGHT_ALIGN_OFFSET), (MODEL_NAME_2_Y + deeppose_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - deeppose_size.width - RIGHT_ALIGN_OFFSET), (MODEL_NAME_2_Y + deeppose_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);

            stream.str("");
            stream << "Pre-Proc: " << fixed << setprecision(2) << PRE_PROC_TIME_DEEPPOSE<<" ms";
            str = stream.str();
            Size pre_proc_size_deeppose = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - pre_proc_size_deeppose.width - RIGHT_ALIGN_OFFSET), (PRE_TIME_STR_Y_GAZE + pre_proc_size_deeppose.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - pre_proc_size_deeppose.width - RIGHT_ALIGN_OFFSET), (PRE_TIME_STR_Y_GAZE + pre_proc_size_deeppose.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);
            stream.str("");
            stream << "Inference: " << fixed << setprecision(2) << INF_TIME_DEEPPOSE<<" ms";
            str = stream.str();
            Size inf_size_deeppose = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - inf_size_deeppose.width - RIGHT_ALIGN_OFFSET), (I_TIME_STR_Y_GAZE + inf_size_deeppose.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - inf_size_deeppose.width - RIGHT_ALIGN_OFFSET), (I_TIME_STR_Y_GAZE + inf_size_deeppose.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);
            stream.str("");
            stream << "Post-Proc: " << fixed << setprecision(2) << POST_PROC_TIME_DEEPPOSE<< " ms";
            str = stream.str();
            Size post_proc_size_deeppose = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - post_proc_size_deeppose.width - RIGHT_ALIGN_OFFSET), (P_TIME_STR_Y_GAZE + post_proc_size_deeppose.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - post_proc_size_deeppose.width - RIGHT_ALIGN_OFFSET), (P_TIME_STR_Y_GAZE + post_proc_size_deeppose.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);

            Size size(DISP_INF_WIDTH, DISP_INF_HEIGHT);
            /*resize the image to the keep ratio size*/
            resize(g_frame, g_frame, size);       

            g_frame.copyTo(output_image(Rect(0, 60, DISP_INF_WIDTH, DISP_INF_HEIGHT)));
            namedWindow("Output Image", WND_PROP_FULLSCREEN);
            setWindowProperty("Output Image", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
            string click_req = "Output Image";
            setMouseCallback(click_req,click_event,NULL);
            cv::Mat bgra_image;
            cv::cvtColor(output_image, bgra_image, cv::COLOR_BGR2BGRA);
            memcpy(img_buffer0, bgra_image.data, DISP_OUTPUT_WIDTH * DISP_OUTPUT_HEIGHT * BGRA_CHANNEL);
            wayland.commit(img_buffer0, NULL);
        }
    }
    free(img_buffer0);
    cap.release(); 
    destroyAllWindows();
    err:
    free(img_buffer0);

    /*Set Termination Request Semaphore to 0*/
    sem_trywait(&terminate_req_sem);
    goto ai_inf_end;
    /*AI Thread Termination*/
    ai_inf_end:
        /*To terminate the loop in Capture Thread.*/
        printf("AI Inference Thread Terminated\n");
        free(img_buffer0);
        pthread_exit(NULL);
        return;
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
    const char* command = "v4l2-ctl --list-devices";
    /* Open a pipe to the command and execute it */ 
    FILE* pipe = popen(command, "r");
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

void *R_Inf_Thread(void *threadid)
{
    int8_t ret = 0;
    ret = wayland.init(DISP_OUTPUT_WIDTH, DISP_OUTPUT_HEIGHT, BGRA_CHANNEL);
    if(0 != ret)
    {
        fprintf(stderr, "[ERROR] Failed to initialize Image for Wayland\n");
        goto err;
    }
    capture_frame(gstreamer_pipeline);

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
    int8_t main_ret = 0;
    /*Semaphore Related*/
    int32_t sem_check = 0;
    /*Variable for checking return value*/
    int8_t ret = 0;

    printf("Main Loop Starts\n");
    while(1)
    {
        /*Gets the Termination request semaphore value. If different then 1 Termination was requested*/
        errno = 0;
        ret = sem_getvalue(&terminate_req_sem, &sem_check);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
            goto err;
        }
        /*Checks the semaphore value*/
        if (1 != sem_check)
        {
            goto main_proc_end;
        }
        /*Wait for 1 TICK.*/
        usleep(WAIT_TIME);
    }

/*Error Processing*/
err:
    sem_trywait(&terminate_req_sem);
    main_ret = 1;
    goto main_proc_end;
/*Main Processing Termination*/
main_proc_end:
    printf("Main Process Terminated\n");
    return main_ret;
}

int main(int argc, char *argv[])
{
    int32_t create_thread_ai = -1;
    int32_t create_thread_key = -1;
    int8_t ret_main = 0;
    int32_t ret = 0;
    int8_t main_proc = 0;
    int32_t sem_create = -1;
    std::string input_source = argv[1];
    std::cout << "Starting Driver Monitoring System Application" << std::endl;
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
                return -1;
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
        return -1;
    }
    
    if (argc>3)
    {
        std::cerr << "[ERROR] Wrong number Arguments are passed " << std::endl;
        return 1;
    }

    errno = 0;
    int drpai_fd = open("/dev/drpai0", O_RDWR);
    if (0 > drpai_fd)
    {
        std::cerr << "[ERROR] Failed to open DRP-AI Driver : errno=" << errno << std::endl;
        return -1;
    }
  
    /*Load Label from label_list file*/
    label_file_map = load_label_file(label_list);

    /*Initialzie DRP-AI (Get DRP-AI memory address and set DRP-AI frequency)*/
    drpaimem_addr_start = init_drpai(drpai_fd);

    if (drpaimem_addr_start == 0)
    {
        close(drpai_fd);
        return -1;
    }

    /*Load model_dir structure and its weight to runtime object */
    runtime_status = runtime.LoadModel(model_dir, drpaimem_addr_start + DRPAI_MEM_OFFSET1);
    
    if(!runtime_status)
    {
        std::cerr << "[ERROR] Failed to load model. " << std::endl;
        close(drpai_fd);
        return -1;
    }    

    std::cout << "[INFO] loaded runtime model :" << model_dir << "\n\n";

     
     /*Load model_dir structure and its weight to runtime object */
    runtime_status1 = runtime1.LoadModel(model_dir1, drpaimem_addr_start + DRPAI_MEM_OFFSET);
    
    if(!runtime_status1)
    {
        std::cerr << "[ERROR] Failed to load model. " << std::endl;
        close(drpai_fd);
        return -1;
    }    

    std::cout << "[INFO] loaded runtime model :" << model_dir1 << "\n\n";
     /* mipi source not supprted */ 

    switch (input_source_map[input_source])
    {
        /* Input Source : USB*/
        case 1:
        {
            std::cout << "[INFO] USB CAMERA \n";
            media_port = query_device_status("usb");
            gstreamer_pipeline = "v4l2src device=" + media_port + " ! video/x-raw, width=640, height=480 ! videoconvert ! appsink";
            sem_create = sem_init(&terminate_req_sem, 0, 1);
            if (0 != sem_create)
            {
                fprintf(stderr, "[ERROR] Failed to Initialize Termination Request Semaphore.\n");
                ret_main = -1;
                goto end_threads;
            }

            create_thread_key = pthread_create(&kbhit_thread, NULL, R_Kbhit_Thread, NULL);
            if (0 != create_thread_key)
            {
                fprintf(stderr, "[ERROR] Failed to create Key Hit Thread.\n");
                ret_main = -1;
                goto end_threads;
            }

            create_thread_ai = pthread_create(&ai_inf_thread, NULL, R_Inf_Thread, NULL);
            if (0 != create_thread_ai)
            {
                sem_trywait(&terminate_req_sem);
                fprintf(stderr, "[ERROR] Failed to create AI Inference Thread.\n");
                ret_main = -1;
                goto end_threads;
            }
        }
        break;
    }

    main_proc = R_Main_Process();
        if (0 != main_proc)
        {
            fprintf(stderr, "[ERROR] Error during Main Process\n");
            ret_main = -1;
        }
        goto end_threads;

 end_threads:

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
    close(drpai_fd);
    return 0;

}
