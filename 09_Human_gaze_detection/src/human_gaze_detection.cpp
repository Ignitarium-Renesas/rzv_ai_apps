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
* File Name    : human_gaze_detection.cpp
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

static Wayland wayland;
static pthread_t ai_inf_thread;
static pthread_t kbhit_thread;
static sem_t terminate_req_sem;
static int32_t drpai_freq;


/*Global Variables*/
static float drpai_output_buf[INF_OUT_SIZE_TINYYOLOV2];
static float drpai_output_buf1[INF_OUT_SIZE_RESNET];

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
float TOTAL_TIME_GAZE = 0;
int32_t HEAD_COUNT= 0;
int fd;

float POST_PROC_TIME_TINYYOLO =0;
float POST_PROC_TIME_GAZE =0;
float PRE_PROC_TIME_TINYYOLO =0;
float PRE_PROC_TIME_GAZE =0;
float INF_TIME_GAZE = 0;
float INF_TIME_TINYYOLO = 0;

/*Cropping parameters*/
static int16_t cropx1[NUM_MAX_FACE];
static int16_t cropy1[NUM_MAX_FACE];
static int16_t cropx2[NUM_MAX_FACE];
static int16_t cropy2[NUM_MAX_FACE];

static uint32_t id_x[2][NUM_MAX_FACE];
static uint32_t id_y[2][NUM_MAX_FACE];

static float resnet18_preds[2][NUM_MAX_FACE];
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
    return 1.0 / (1.0 + exp(-x));
}

/*****************************************
* Function Name : softmax
* Description   : Helper function for YOLO Post Processing
* Arguments     : val[] = array to be computed softmax
* Return value  : -
******************************************/
static void softmax(float val[NUM_CLASS])
{
    float max_num = -FLT_MAX;
    float sum = 0;
    int32_t i;
    for ( i = 0 ; i<NUM_CLASS ; i++ )
    {
        max_num = std::max(max_num, val[i]);
    }

    for ( i = 0 ; i<NUM_CLASS ; i++ )
    {
        val[i]= (float) exp(val[i] - max_num);
        sum+= val[i];
    }

    for ( i = 0 ; i<NUM_CLASS ; i++ )
    {
        val[i]= val[i]/sum;
    }
    return;
}


/*****************************************
* Function Name : index
* Description   : Get the index of the bounding box attributes based on the input offset
* Arguments     : offs = offset to access the bounding box attributes
*                 channel = channel to access each bounding box attribute.
* Return value  : index to access the bounding box attribute.
******************************************/
static int32_t index(int32_t offs, int32_t channel)
{
    return offs + channel * NUM_GRID_X * NUM_GRID_Y;
}

/*****************************************
* Function Name : offset_yolo
* Description   : Get the offset number to access the bounding box attributes
*                 To get the actual value of bounding box attributes, use index() after this function.
* Arguments     : b = Number to indicate which bounding box in the region [0~4]
*                 y = Number to indicate which region [0~13]
*                 x = Number to indicate which region [0~13]
* Return value  : offset to access the bounding box attributes.
*******************************************/
static int offset_yolo(int b, int y, int x)
{
    return b *(NUM_CLASS + 5)* NUM_GRID_X * NUM_GRID_Y + y * NUM_GRID_X + x;
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


 void R_Post_Proc_ResNet18(float* floatarr, uint8_t n_pers)
{
    resnet18_preds[0][n_pers] = floatarr[0];
    resnet18_preds[1][n_pers] = floatarr[1];
    if (resnet18_preds[1][n_pers]>0)
    return;
}


void R_ResNet18_Coord_Convert(uint8_t n_pers)
{
    /* Render skeleton on image and print their details */
    int32_t x1, x2;
    int32_t y1, y2;
    int32_t dx, dy;
    int32_t length = (cropx2[n_pers]-cropx1[n_pers]);
    int32_t height = (cropy2[n_pers]-cropy1[n_pers]);

    /* Conversion from input image coordinates to display image coordinates. */
    /* Make sure the coordinates are not off the screen. */
    x1 = (int32_t)(cropx1[n_pers] + length/2);
    y1 = (int32_t)(cropy1[n_pers] + height/4);
    x1 = (x1 < 0) ? 0 : x1;
    x1 = (x1 > DRPAI_IN_WIDTH - KEY_POINT_SIZE -1 ) ? DRPAI_IN_WIDTH -KEY_POINT_SIZE -1 : x1;
    y1 = (y1 < 0) ? 0 : y1;
    y1 = (y1 > DRPAI_IN_HEIGHT -KEY_POINT_SIZE -1) ? DRPAI_IN_HEIGHT -KEY_POINT_SIZE -1 : y1;

    id_x[0][n_pers] = x1;
    id_y[0][n_pers] = y1;

    dx = (int32_t) -1 * length/2 * cos(resnet18_preds[0][n_pers]) * sin(resnet18_preds[1][n_pers]);
    dy = (int32_t) -1 * length/2 * sin(resnet18_preds[0][n_pers]);
    x2 = (int32_t) x1 + dx;
    y2 = (int32_t) y1 + dy;
    x2 = (x2 < 0) ? 0 : x2;
    x2 = (x2 > DRPAI_IN_WIDTH - KEY_POINT_SIZE -1 ) ? DRPAI_IN_WIDTH -KEY_POINT_SIZE -1 : x2;
    y2 = (y2 < 0) ? 0 : y2;
    y2 = (y2 > DRPAI_IN_HEIGHT -KEY_POINT_SIZE -1) ? DRPAI_IN_HEIGHT -KEY_POINT_SIZE -1 : y2;

    id_x[1][n_pers] = x2;
    id_y[1][n_pers] = y2;
    return;
}

/*****************************************
 * Function Name : R_Post_Proc
 * Description   : Process CPU post-processing for YOLOv3
 * Arguments     : floatarr = drpai output address
 * Return value  : -
 ******************************************/
void R_Post_Proc(float *floatarr)
{
    /* Following variables are required for correct_region_boxes in Darknet implementation*/
    /* Note: This implementation refers to the "darknet detector test" */
    
    float new_w, new_h;
    int32_t result_cnt =0;
    int32_t count =0;
    float correct_w = 1.;
    float correct_h = 1.;
    if ((float)(MODEL_IN_W / correct_w) < (float)(MODEL_IN_H / correct_h))
    {
        new_w = (float)MODEL_IN_W;
        new_h = correct_h * MODEL_IN_W / correct_w;
    }
    else
    {
        new_w = correct_w * MODEL_IN_H / correct_h;
        new_h = MODEL_IN_H;
    }

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
    Box bb;
    float classes[NUM_CLASS];
    float max_pred = 0;
    int32_t pred_class = -1;
    float probability = 0;
    detection d;
    /* Clear the detected result list */
    det.clear();

    /*Post Processing Start*/
    for(b = 0; b < NUM_BB; b++)
    {
        for(y = 0; y < NUM_GRID_Y; y++)
        {
            for(x = 0; x < NUM_GRID_X; x++)
            {
                offs = offset_yolo(b, y, x);
                tx = floatarr[offs];
                ty = floatarr[index(offs, 1)];
                tw = floatarr[index(offs, 2)];
                th = floatarr[index(offs, 3)];
                tc = floatarr[index(offs, 4)];

                /* Compute the bounding box */
                /*get_region_box*/
                center_x = ((float) x + sigmoid(tx)) / (float) NUM_GRID_X;
                center_y = ((float) y + sigmoid(ty)) / (float) NUM_GRID_Y;
                box_w = (float) exp(tw) * anchors[2*b+0] / (float) NUM_GRID_X;
                box_h = (float) exp(th) * anchors[2*b+1] / (float) NUM_GRID_Y;

                /* Adjustment for VGA size */
                /* correct_region_boxes */
                center_x = (center_x - (MODEL_IN_W - new_w) / 2. / MODEL_IN_W) / ((float) new_w / MODEL_IN_W);
                center_y = (center_y - (MODEL_IN_H - new_h) / 2. / MODEL_IN_H) / ((float) new_h / MODEL_IN_H);
                box_w *= (float) (MODEL_IN_W / new_w);
                box_h *= (float) (MODEL_IN_H / new_h);

                center_x = round(center_x * DRPAI_IN_WIDTH);
                center_y = round(center_y * DRPAI_IN_HEIGHT);
                box_w = round(box_w * DRPAI_IN_WIDTH);
                box_h = round(box_h * DRPAI_IN_HEIGHT);

                objectness = sigmoid(tc);

                bb = {center_x, center_y, box_w, box_h};
                /* Get the class prediction */
                for (i = 0; i < NUM_CLASS; i++)
                {
                    classes[i] = floatarr[index(offs, 5+i)];
                }
                softmax(classes);
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
                if ((probability > TH_PROB))
                {
              
                    d = {bb, pred_class, probability};
                    det.push_back(d);
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
            if(count > 2)
            {
                break;
            }
        }
    }
    HEAD_COUNT = result_cnt++;
    return;
}

/*****************************************
 * Function Name : draw_bounding_box
 * Description   : Draw bounding box on image.
 * Arguments     : -
 * Return value  : 0 if succeeded
 *               not 0 otherwise
 ******************************************/
void draw_bounding_box(void)
{
    stringstream stream;
    string str = "";
    string result_str;
    int32_t result_cnt =0;
    uint32_t x = HEAD_COUNT_STR_X;
    uint32_t y = HEAD_COUNT_STR_X;
    /* Draw bounding box on RGB image. */
    int32_t i = 0;
    for (i = 0; i < HEAD_COUNT; i++)
    {
        /* Skip the overlapped bounding boxes */
        if (det[i].prob == 0)
        {
            continue;
        }
        result_cnt++;
        /* Clear string stream for bounding box labels */
        stream.str("");
        /* Draw the bounding box on the image */
        stream << fixed << setprecision(2) << det[i].prob;
        result_str = label_file_map[det[i].c] + " " + stream.str();    
        
        int32_t x2_min = cropx1[i] + BOX_THICKNESS;
        int32_t y2_min = cropy1[i] + BOX_THICKNESS;
        int32_t x2_max = cropx2[i] - BOX_THICKNESS;
        int32_t y2_max = cropy2[i] - BOX_THICKNESS;

        int32_t height = (y2_max - y2_min)*1.6;
        int32_t width = (x2_max - x2_min)*1.6;

        x2_min = (x2_min + x2_max)/2 - width/2;
        x2_max = (x2_min + x2_max)/2 + width/2;
        y2_max = (y2_min + y2_max)/2 + height/2;
        y2_min = (y2_min + y2_max)/2 - height/2;
        
        x2_min = ((DRPAI_IN_WIDTH - 2) < x2_min) ? (DRPAI_IN_WIDTH - 2) : x2_min;
        x2_max = x2_max < 1 ? 1 : x2_max;
        y2_min = ((DRPAI_IN_HEIGHT - 2) < y2_min) ? (DRPAI_IN_HEIGHT - 2) : y2_min;
        y2_max = y2_max < 1 ? 1 : y2_max;


        Point topLeft2(x2_min, y2_min);
        Point bottomRight2(x2_max, y2_max);

        /* Creating bounding box and class labels */
        /*cordinates for solid rectangle*/

        rectangle(g_frame, topLeft2, bottomRight2, Scalar(0, 255, 0), BOX_THICKNESS);
        
        /* gaze */
        Point topLeftgaze(id_x[0][i],id_y[0][i]);
        circle(g_frame, topLeftgaze ,2, Scalar(255, 255, 255), -1);
        Point topRightgaze(id_x[1][i],id_y[1][i]);
        line( g_frame,topLeftgaze,topRightgaze, Scalar( 0, 255, 0 ),1, LINE_8 );
        circle(g_frame, topRightgaze ,2, Scalar(255, 0, 0), -1);
        
    }
    return;
}


/*****************************************
 * Function Name : Gaze Detection
 * Description   : Function to perform over all detection
 * Arguments     : -
 * Return value  : 0 if succeeded
 *               not 0 otherwise
 ******************************************/
int Gaze_Detection()
{   
    int wait_key;
     /* Temp frame */
    Mat frame1;

    Size size(MODEL_IN_H, MODEL_IN_W);
    /*Pre process start time for tinyyolo model */
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

    if (HEAD_COUNT > 0){

       float POST_PROC_TIME_GAZE_MICRO =0;
       float PRE_PROC_TIME_GAZE_MICRO =0;
       float INF_TIME_GAZE_MICRO = 0;
        
        for (int i = 0 ; i < HEAD_COUNT; i++)
        {
        
            /* Preprocess time start for gaze model*/
            auto t0_gaze = std::chrono::high_resolution_clock::now();

            cropx1[i] = (int)det[i].bbox.x - round((int)det[i].bbox.w / 2.);
            cropy1[i] = (int)det[i].bbox.y - round((int)det[i].bbox.h / 2.);
            cropx2[i] = (int)det[i].bbox.x + round((int)det[i].bbox.w / 2.) - 1;
            cropy2[i] = (int)det[i].bbox.y + round((int)det[i].bbox.h / 2.) - 1;

            /* Check the bounding box is in the image range */
            cropx1[i] = cropx1[i] < 1 ? 1 : cropx1[i];
            cropx2[i] = ((DRPAI_IN_WIDTH - 2) < cropx2[i]) ? (DRPAI_IN_WIDTH - 2) : cropx2[i];
            cropy1[i] = cropy1[i] < 1 ? 1 : cropy1[i];
            cropy2[i] = ((DRPAI_IN_HEIGHT - 2) < cropy2[i]) ? (DRPAI_IN_HEIGHT - 2) : cropy2[i];

            Mat cropped_image = g_frame(Range(cropy1[i],cropy2[i]), Range(cropx1[i],cropx2[i]));
            
            Mat frame1res;

            Size size(224, 224);

            /*resize the image to the model input size*/
            resize(cropped_image, frame1res, size);
            vector<Mat> rgb_imagesres;
            split(frame1res, rgb_imagesres);
            Mat m_flat_r_res = rgb_imagesres[0].reshape(1, 1);
            Mat m_flat_g_res = rgb_imagesres[1].reshape(1, 1);
            Mat m_flat_b_res = rgb_imagesres[2].reshape(1, 1);
            Mat matArrayres[] = {m_flat_r_res, m_flat_g_res, m_flat_b_res};
            Mat frameCHWres;
            hconcat(matArrayres, 3, frameCHWres);
            /*convert to FP32*/
            frameCHWres.convertTo(frameCHWres, CV_32FC3);

            /* normailising  pixels */
            divide(frameCHWres, 255.0, frameCHWres);

            /* DRP AI input image should be continuous buffer */
            if (!frameCHWres.isContinuous())
                frameCHWres = frameCHWres.clone();

            Mat frameres = frameCHWres;
            
            auto t1_gaze = std::chrono::high_resolution_clock::now();

            /* resnet18 inference*/
            runtime1.SetInput(0, frameres.ptr<float>());
            /*inference start time for gaze model*/
            auto t2_gaze = std::chrono::high_resolution_clock::now();
            runtime1.Run(drpai_freq);
            /*inference end time for gaze model*/
            auto t3_gaze = std::chrono::high_resolution_clock::now();
            auto inf_duration_gaze = std::chrono::duration_cast<std::chrono::microseconds>(t3_gaze - t2_gaze).count();
        
        /*Postprocess time start for gaze model*/
            auto t4_gaze = std::chrono::high_resolution_clock::now();
            /*load inference out on drpai_out_buffer*/
            int32_t l = 0;
            int32_t output_num_res = 0;
            std::tuple<InOutDataType, void *, int64_t> output_buffer_res;
            int64_t output_size_res;
            uint32_t size_count_res = 0;

            /* Get the number of output of the target model. */
            output_num_res = runtime1.GetNumOutput();
            size_count_res = 0;
            /*GetOutput loop*/
            for (l = 0; l < output_num_res; l++)
            {
                /* output_buffer below is tuple, which is { data type, address of output data, number of elements } */
                output_buffer_res = runtime1.GetOutput(l);
                /*Output Data Size = std::get<2>(output_buffer). */
                output_size_res = std::get<2>(output_buffer_res);

                /*Output Data Type = std::get<0>(output_buffer)*/
                if (InOutDataType::FLOAT16 == std::get<0>(output_buffer_res))
                {
                    /*Output Data = std::get<1>(output_buffer)*/
                    uint16_t *data_ptr_res = reinterpret_cast<uint16_t *>(std::get<1>(output_buffer_res));
                    for (int j = 0; j < output_size_res; j++)
                    {
                        /*FP16 to FP32 conversion*/
                        drpai_output_buf1[j + size_count_res] = float16_to_float32(data_ptr_res[j]);
                    }
                }
                else if (InOutDataType::FLOAT32 == std::get<0>(output_buffer_res))
                {
                    /*Output Data = std::get<1>(output_buffer)*/
                    float *data_ptr_res = reinterpret_cast<float *>(std::get<1>(output_buffer_res));
                    for (int j = 0; j < output_size_res; j++)
                    {
                        drpai_output_buf1[j + size_count_res] = data_ptr_res[j];
                    }
                }
                else
                {
                    std::cerr << "[ERROR] Output data type : not floating point." << std::endl;
                    ret = -1;
                    break;
                }
                size_count_res += output_size_res;
            }

            if (ret != 0)
            {
                std::cerr << "[ERROR] DRP Inference Not working !!! " << std::endl;
                return -1;
            }
            /*Post process start time for gaze model*/
            R_Post_Proc_ResNet18(drpai_output_buf1, i);
            R_ResNet18_Coord_Convert(i);
            
            /*Postprocess time end for gaze model*/
            auto t5_gaze = std::chrono::high_resolution_clock::now();

            auto r_post_proc_time_gaze = std::chrono::duration_cast<std::chrono::microseconds>(t5_gaze - t4_gaze).count();
            auto pre_proc_time_gaze = std::chrono::duration_cast<std::chrono::microseconds>(t1_gaze - t0_gaze).count();
            
            POST_PROC_TIME_GAZE_MICRO = POST_PROC_TIME_GAZE_MICRO + r_post_proc_time_gaze;
            PRE_PROC_TIME_GAZE_MICRO = PRE_PROC_TIME_GAZE_MICRO + pre_proc_time_gaze;
            INF_TIME_GAZE_MICRO = INF_TIME_GAZE_MICRO + inf_duration_gaze;

        }
        POST_PROC_TIME_GAZE = POST_PROC_TIME_GAZE_MICRO/1000.0;
        PRE_PROC_TIME_GAZE = PRE_PROC_TIME_GAZE_MICRO/1000.0;
        INF_TIME_GAZE = INF_TIME_GAZE_MICRO/1000.0;
        float total_time = float(POST_PROC_TIME_GAZE_MICRO) + float(PRE_PROC_TIME_GAZE_MICRO) + float(INF_TIME_GAZE_MICRO);
        TOTAL_TIME_GAZE = total_time/1000.0;

    }
    auto r_post_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
    auto pre_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    POST_PROC_TIME_TINYYOLO = r_post_proc_time/1000.0;
    PRE_PROC_TIME_TINYYOLO = pre_proc_time/1000.0;
    INF_TIME_TINYYOLO = inf_duration/1000.0;

    float total_time_tinyyolo = float(POST_PROC_TIME_TINYYOLO) + float(PRE_PROC_TIME_TINYYOLO) + float(INF_TIME_TINYYOLO);
    TOTAL_TIME = TOTAL_TIME_GAZE + total_time_tinyyolo;

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
    int32_t ret = 0;
    int32_t baseline = 10;
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
            int ret = Gaze_Detection();
            if (ret != 0)
            {
                std::cerr << "[ERROR] Inference Not working !!! " << std::endl;
            }

            /* Draw bounding box on the frame */
            draw_bounding_box();
            /*Display frame */
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
            stream << "TinyYolov2";
            str = stream.str();
            Size tinyyolov2_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - tinyyolov2_size.width - RIGHT_ALIGN_OFFSET), (MODEL_NAME_1_Y + tinyyolov2_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - tinyyolov2_size.width - RIGHT_ALIGN_OFFSET), (MODEL_NAME_1_Y + tinyyolov2_size.height)), FONT_HERSHEY_SIMPLEX, 
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
            
            /*Gaze model Timings*/
            stream.str("");
            stream << "Resnet18";
            str = stream.str();
            Size resnet18_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - resnet18_size.width - RIGHT_ALIGN_OFFSET), (MODEL_NAME_2_Y + resnet18_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - resnet18_size.width - RIGHT_ALIGN_OFFSET), (MODEL_NAME_2_Y + resnet18_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);

            stream.str("");
            stream << "Pre-Proc: " << fixed << setprecision(2) << PRE_PROC_TIME_GAZE<<" ms";
            str = stream.str();
            Size pre_proc_size_gaze = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - pre_proc_size_gaze.width - RIGHT_ALIGN_OFFSET), (PRE_TIME_STR_Y_GAZE + pre_proc_size_gaze.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - pre_proc_size_gaze.width - RIGHT_ALIGN_OFFSET), (PRE_TIME_STR_Y_GAZE + pre_proc_size_gaze.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);
            stream.str("");
            stream << "Inference: " << fixed << setprecision(2) << INF_TIME_GAZE<<" ms";
            str = stream.str();
            Size inf_size_gaze = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - inf_size_gaze.width - RIGHT_ALIGN_OFFSET), (I_TIME_STR_Y_GAZE + inf_size_gaze.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - inf_size_gaze.width - RIGHT_ALIGN_OFFSET), (I_TIME_STR_Y_GAZE + inf_size_gaze.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);
            stream.str("");
            stream << "Post-Proc: " << fixed << setprecision(2) << POST_PROC_TIME_GAZE << " ms";
            str = stream.str();
            Size post_proc_size_gaze = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - post_proc_size_gaze.width - RIGHT_ALIGN_OFFSET), (P_TIME_STR_Y_GAZE + post_proc_size_gaze.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - post_proc_size_gaze.width - RIGHT_ALIGN_OFFSET), (P_TIME_STR_Y_GAZE + post_proc_size_gaze.height)), FONT_HERSHEY_SIMPLEX, 
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
    std::cout << "Starting Human gaze Application" << std::endl;

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
    if (argc>5)
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
