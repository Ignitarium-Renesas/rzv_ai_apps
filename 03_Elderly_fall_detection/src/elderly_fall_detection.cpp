
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
* File Name    : elderly_fall_detection.cpp
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
static float drpai_output_buf[INF_OUT_SIZE_TINYYOLOV2];
static float drpai_output_buf1[INF_OUT_SIZE_HRNET];
static Wayland wayland;
static pthread_t ai_inf_thread;
static pthread_t kbhit_thread;
static sem_t terminate_req_sem;
static int32_t drpai_freq;
static atomic<uint8_t> hdmi_obj_ready   (0);
static uint32_t disp_time = 0;
std::string media_port;
std::string gstreamer_pipeline;

uint64_t drpaimem_addr_start = 0;
bool runtime_status = false; 
bool runtime_status1 = false; 

static int16_t cropx1[NUM_MAX_PERSON];
static int16_t cropy1[NUM_MAX_PERSON];
static int16_t cropx2[NUM_MAX_PERSON];
static int16_t cropy2[NUM_MAX_PERSON];
static float hrnet_preds[NUM_OUTPUT_C][3];
static uint16_t id_x[NUM_OUTPUT_C][NUM_MAX_PERSON]; 
static uint16_t id_y[NUM_OUTPUT_C][NUM_MAX_PERSON]; 

static int16_t cropx[NUM_MAX_PERSON];
static int16_t cropy[NUM_MAX_PERSON];
static int16_t croph[NUM_MAX_PERSON];
static int16_t cropw[NUM_MAX_PERSON];
static float lowest_kpt_score_local[NUM_MAX_PERSON]; 
static uint8_t fall_frame_count[NUM_MAX_PERSON] = {0};
static int8_t flag[NUM_MAX_PERSON] = {0};

/*TinyYOLOv2*/
static uint32_t person_cnt = 0;
static std::vector<detection> det;
static std::vector<detection> det_ppl;

float fps = 0;
float TOTAL_TIME = 0;
float TOTAL_TIME_HRNET = 0;
float POST_PROC_TIME_TINYYOLO =0;
float POST_PROC_TIME_HRNET =0;
float PRE_PROC_TIME_TINYYOLO =0;
float PRE_PROC_TIME_HRNET =0;
float INF_TIME_HRNET = 0;
float INF_TIME_TINYYOLO = 0;


/*Global frame */
Mat g_frame;
VideoCapture cap;

cv::Mat output_image;

/* Map to store input source list */
std::map<std::string, int> input_source_map ={  {"USB", 1} } ;


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
static double sigmoid(double x)
{
    return 1.0/(1.0+exp(-x));
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
* Function Name : R_Post_Proc
* Description   : Process CPU post-processing for TinyYOLOv2
* Arguments     : address = drpai output address
*                 det = detected boxes details
*                 box_count = total number of boxes
* Return value  : -
******************************************/
static void R_Post_Proc(float *floatarr)
{
    /* Following variables are required for correct_yolo_boxes in Darknet implementation*/
    /* Note: This implementation refers to the "darknet detector test" */
    float new_w, new_h;
    float correct_w = 1.;
    float correct_h = 1.;
    if ((float)(MODEL_IN_W / correct_w) < (float)(MODEL_IN_H/correct_h) )
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
    det_ppl.clear();
    person_cnt=0;

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
                if (probability > TH_PROB)   
                {
                    d = {bb, pred_class, probability};
                    det.push_back(d);
                }
            }
        }
    }
    /* Non-Maximum Supression filter */
    filter_boxes_nms(det, det.size(), TH_NMS);
    if(det.size()>0)
    {
        for (i = 0; i < det.size(); i++)
        {
            /* Skip the overlapped bounding boxes */
            if (det[i].prob == 0)
            {
                continue;
            }
            else
            {
                det_ppl.push_back(det[i]);
                person_cnt++;
                if(person_cnt == NUM_MAX_PERSON)
                {
                    break;
                }
            }
        }
    }
    else
    {
        person_cnt=0;
    }
    return;
}

/*****************************************
* Function Name : offset_hrnet
* Description   : Get the offset number to access the HRNet attributes
* Arguments     : b = Number to indicate which region [0~17]
*                 y = Number to indicate which region [0~64]
*                 x = Number to indicate which region [0~48]
* Return value  : offset to access the HRNet attributes.
*******************************************/
static int32_t offset_hrnet(int32_t b, int32_t y, int32_t x)
{
    return b * NUM_OUTPUT_W * NUM_OUTPUT_H + y * NUM_OUTPUT_W + x;
}

/*****************************************
* Function Name : sign
* Description   : Get the sign of the input value
* Arguments     : x = input value
* Return value  : returns the sign, 1 if positive -1 if not
*******************************************/
static int8_t sign(int32_t x)
{
    return x > 0 ? 1 : -1;
}

/*****************************************
* Function Name : R_Post_Proc_HRNet
* Description   : CPU post-processing for HRNet
                  MPII Human Pose
*                 More details can be found in the 
*                 <http://human-pose.mpi-inf.mpg.de/#overview>
*                 MPII Keypoint Indexes:
                    0 - r ankle,
                    1 - r knee, 
                    2 - r hip, 
                    3 - l hip,
                    4 - l knee,
                    5 - l ankle, 
                    6 - pelvis, 
                    7 - thorax, 
                    8 - upper neck, 
                    9 - head top, 
                    10 - r wrist, 
                    11 - r elbow, 
                    12 - r shoulder, 
                    13 - l shoulder, 
                    14 - l elbow, 
                    15 - l wrist
* Arguments     : floatarr = drpai output address
*                 n_pers = number of the person detected
* Return value  : -
******************************************/
static void R_Post_Proc_HRNet(float* floatarr, uint8_t n_pers)
{
    float score;
    int32_t b = 0;
    int32_t y = 0;
    int32_t x = 0;
    int32_t offs = 0;

    float center[] = {(float)(cropw[n_pers] / 2 -1), (float)(croph[n_pers] / 2 - 1)};
    int8_t ind_x = -1;
    int8_t ind_y = -1;
    float max_val = -1;
    float scale_x = 0;
    float scale_y = 0;
    float coords_x = 0;
    float coords_y = 0;
    float diff_x;
    float diff_y;
    int8_t i;
    
    for(b = 0; b < NUM_OUTPUT_C; b++)
    {
        float scale[] = {(float)(cropw[n_pers] / 200.0), (float)(croph[n_pers] / 200.0)};
        ind_x = -1;
        ind_y = -1;
        max_val = -1;
        for(y = 0; y < NUM_OUTPUT_H; y++)
        {
            for(x = 0; x < NUM_OUTPUT_W; x++)
            {
                offs = offset_hrnet(b, y, x);
                if (max_val < floatarr[offs])
                {
                    /*Update the maximum value and indices*/
                    max_val = floatarr[offs];
                    ind_x = x;
                    ind_y = y;
                }
            }
        }
        if (0 > max_val)
        {
            ind_x = -1;
            ind_y = -1;
            goto not_detect;
        }
        hrnet_preds[b][0] = float(ind_x);
        hrnet_preds[b][1] = float(ind_y);
        hrnet_preds[b][2] = max_val;
        offs = offset_hrnet(b, ind_y, ind_x);
        if ((ind_y > 1) && (ind_y < NUM_OUTPUT_H -1))
        {
            if ((ind_x > 1) && (ind_x < (NUM_OUTPUT_W -1)))
            {
                diff_x = floatarr[offs + 1] - floatarr[offs - 1];
                diff_y = floatarr[offs + NUM_OUTPUT_W] - floatarr[offs - NUM_OUTPUT_W];
                hrnet_preds[b][0] += sign(diff_x) * 0.25;
                hrnet_preds[b][1] += sign(diff_y) * 0.25;
            }
        }

        /*transform_preds*/
        scale[0] *= 200;
        scale[1] *= 200;
        scale_x = scale[0] / (NUM_OUTPUT_W);
        scale_y = scale[1] / (NUM_OUTPUT_H);
        coords_x = hrnet_preds[b][0];
        coords_y = hrnet_preds[b][1];
        hrnet_preds[b][0] = (coords_x * scale_x) + center[0] - (scale[0] * 0.5);
        hrnet_preds[b][1] = (coords_y * scale_y) + center[1] - (scale[1] * 0.5);
    }
    /* Clear the score in preparation for the update. */
    lowest_kpt_score_local[n_pers] = 0;
    score = 1;
    for (i = 0; i < NUM_OUTPUT_C; i++)
    {
        /* Adopt the lowest score. */
        if (hrnet_preds[i][2] < score)
        {
            score = hrnet_preds[i][2];
        }
    }
    /* Update the score for display thread. */
    lowest_kpt_score_local[n_pers] = score;
    goto end;

not_detect:
    lowest_kpt_score_local[n_pers] = 0;
    goto end;

end:
    return;
}

/*****************************************
* Function Name : R_HRNet_Coord_Convert
* Description   : Convert the post processing result into drawable coordinates
* Arguments     : n_pers = number of the detected person
* Return value  : -
******************************************/
static void R_HRNet_Coord_Convert(uint8_t n_pers)
{
    /* Render skeleton on image and print their details */
    int32_t posx;
    int32_t posy;
    int8_t i;

    for (i = 0; i < NUM_OUTPUT_C; i++)
    {
        /* Conversion from input image coordinates to display image coordinates. */
        /* +0.5 is for rounding.*/
        posx = (int32_t)(hrnet_preds[i][0] + 0.5) + cropx[n_pers] + OUTPUT_ADJ_X;
        posy = (int32_t)(hrnet_preds[i][1] + 0.5) + cropy[n_pers] + OUTPUT_ADJ_Y;
        /* Make sure the coordinates are not off the screen. */
        posx = (posx < 0) ? 0 : posx;
        posx = (posx > IMAGE_WIDTH - KEY_POINT_SIZE -1 ) ? IMAGE_WIDTH -KEY_POINT_SIZE -1 : posx;
        posy = (posy < 0) ? 0 : posy;
        posy = (posy > IMAGE_HEIGHT -KEY_POINT_SIZE -1) ? IMAGE_HEIGHT -KEY_POINT_SIZE -1 : posy;
        id_x[i][n_pers] = posx;
        id_y[i][n_pers] = posy;
    }
    return;
}

/*****************************************
* Function Name : draw_skeleton
* Description   : Draw Complete Skeleton on image.
* Arguments     : -
* Return value  : -
******************************************/
static void draw_skeleton(void)
{
    uint8_t i, j ,v ,count=0;
    int16_t x_diff = 0;
    int16_t y_diff = 0;
    
    if(person_cnt > 0)
    {
        for(i=0; i < person_cnt; i++)
        {   
            /*Check If All Key Points Were Detected: If Over Threshold, It will Draw Complete Skeleton*/
            if (lowest_kpt_score_local[i] > TH_KPT)
            {
                x_diff = abs(id_x[13][i] - id_x[5][i]);
                y_diff = abs(id_y[13][i] - id_y[5][i]);
 
                /*Draw Rectangle As Key Points*/
                for(v = 0; v < NUM_OUTPUT_C; v++)
                {
                    if(v==13 || v==12 || v==5 || v==0)
                    {
                    /*Draw Rectangles On Shoulders ,ankles Key Points*/
                    rectangle(g_frame, Point(id_x[v][i], id_y[v][i]), Point(id_x[v][i], id_y[v][i]), Scalar(0, 0, 255), KEY_POINT_SIZE);
                    }
  
                }
                if(x_diff > y_diff || (cropw[i]/croph[i])>0.5)
                {
                    flag[i] = 1;
                    if((fall_frame_count[i] <= 5))
                        {
                            fall_frame_count[i]++;
                        }
                }
                else{
                fall_frame_count[i] = 0;
                flag[i] = 0;
                }
            }
            else if ((cropw[i]/croph[i])>0.5)
            {
                flag[i] = 1;
                if((fall_frame_count[i] <= 5))
                       { 
                        fall_frame_count[i]++;
                       }
            }
            else
            {
                fall_frame_count[i] = 0;
                flag[i] = 0;
            }
        }
    }
    else
    {
        fall_frame_count[i] = 0;
        flag[i] = 0;
    }
    
    return;
}


/*****************************************
 * Function Name : Fall Detection
 * Description   : Function to perform over all detection
 * Arguments     : -
 * Return value  : 0 if succeeded
 *               not 0 otherwise
 ******************************************/
int Fall_Detection()
{   
    /* Temp frame */
    Mat frame1;

    Size size(MODEL_IN_H, MODEL_IN_W);

    /* Preprocess time for tinyyolo model start */
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

    /*start inference using drp runtime*/
    runtime.SetInput(0, frame.ptr<float>());

    /* Inference time start for tinyyolo model */
    auto t2 = std::chrono::high_resolution_clock::now();
    runtime.Run(drpai_freq);
    /* Inference time end for tinyyolo model*/
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
    /* Postprocess time end for tinyyolo model*/
    auto t5 = std::chrono::high_resolution_clock::now();

    if ( person_cnt > 0)
    {
       float POST_PROC_TIME_HRNET_MICRO =0;
       float PRE_PROC_TIME_HRNET_MICRO =0;
       float INF_TIME_HRNET_MICRO = 0;
        
        for (int k = 0 ; k < person_cnt; k++)
        {
        
            /* Preprocess time start for hrnet model*/
            auto t0_hrnet = std::chrono::high_resolution_clock::now();

            cropx1[k] = (int)det_ppl[k].bbox.x - round((int)det_ppl[k].bbox.w / 2.);
            cropy1[k] = (int)det_ppl[k].bbox.y - round((int)det_ppl[k].bbox.h / 2.);
            cropx2[k] = (int)det_ppl[k].bbox.x + round((int)det_ppl[k].bbox.w / 2.) - 1;
            cropy2[k] = (int)det_ppl[k].bbox.y + round((int)det_ppl[k].bbox.h / 2. ) - 1;

            /* Check the bounding box is in the image range */
            cropx1[k] = cropx1[k] < 1 ? 1 : cropx1[k];
            cropx2[k] = ((DRPAI_IN_WIDTH - 2) < cropx2[k]) ? ( DRPAI_IN_WIDTH- 2) : cropx2[k];
            cropy1[k] = cropy1[k] < 1 ? 1 : cropy1[k];
            cropy2[k] = ((DRPAI_IN_HEIGHT - 2) < cropy2[k]) ? (DRPAI_IN_HEIGHT - 2) : cropy2[k];

            Mat cropped_image = g_frame(Range(cropy1[k],cropy2[k]), Range(cropx1[k],cropx2[k]));
            
            Mat frame1_hrnet;

            Size size(MODEL_IN_H_HRNET, MODEL_IN_W_HRNET);

            /*resize the image to the model input size*/
            resize(cropped_image, frame1_hrnet, size);
            vector<Mat> rgb_images_hrnet;
            split(frame1_hrnet, rgb_images_hrnet);
            Mat m_flat_r_hrnet = rgb_images_hrnet[0].reshape(1, 1);
            Mat m_flat_g_hrnet = rgb_images_hrnet[1].reshape(1, 1);
            Mat m_flat_b_hrnet = rgb_images_hrnet[2].reshape(1, 1);
            Mat matArray_hrnet[] = {m_flat_r_hrnet, m_flat_g_hrnet, m_flat_b_hrnet};
            Mat frameCHW_hrnet;
            hconcat(matArray_hrnet, 3, frameCHW_hrnet);
            /*convert to FP32*/
            frameCHW_hrnet.convertTo(frameCHW_hrnet, CV_32FC3);

            /* normailising  pixels */
            divide(frameCHW_hrnet, 255.0, frameCHW_hrnet);

            /* DRP AI input image should be continuous buffer */
            if (!frameCHW_hrnet.isContinuous())
                frameCHW_hrnet = frameCHW_hrnet.clone();

            Mat frame_hrnet = frameCHW_hrnet;
            
            auto t1_hrnet = std::chrono::high_resolution_clock::now();

            /* hrnet inference*/
            runtime1.SetInput(0, frame_hrnet.ptr<float>());
            /*inference start time for hrnet model*/
            auto t2_hrnet = std::chrono::high_resolution_clock::now();
            runtime1.Run(drpai_freq);
            /*inference end time for hrnet model*/
            auto t3_hrnet = std::chrono::high_resolution_clock::now();
            auto inf_duration_hrnet = std::chrono::duration_cast<std::chrono::microseconds>(t3_hrnet - t2_hrnet).count();
        
            /*Postprocess time start for hrnet model*/
            auto t4_hrnet = std::chrono::high_resolution_clock::now();
            /*load inference out on drpai_out_buffer*/
            int32_t l = 0;
            int32_t output_num_hrnet = 0;
            std::tuple<InOutDataType, void *, int64_t> output_buffer_hrnet;
            int64_t output_size_hrnet;
            uint32_t size_count_hrnet = 0;

            /* Get the number of output of the target model. */
            output_num_hrnet = runtime1.GetNumOutput();
            size_count_hrnet = 0;
            /*GetOutput loop*/
            for (l = 0; l < output_num_hrnet; l++)
            {
                /* output_buffer below is tuple, which is { data type, address of output data, number of elements } */
                output_buffer_hrnet = runtime1.GetOutput(l);
                /*Output Data Size = std::get<2>(output_buffer). */
                output_size_hrnet = std::get<2>(output_buffer_hrnet);

                /*Output Data Type = std::get<0>(output_buffer)*/
                if (InOutDataType::FLOAT16 == std::get<0>(output_buffer_hrnet))
                {
                    /*Output Data = std::get<1>(output_buffer)*/
                    uint16_t *data_ptr_hrnet = reinterpret_cast<uint16_t *>(std::get<1>(output_buffer_hrnet));
                    for (int j = 0; j < output_size_hrnet; j++)
                    {
                        /*FP16 to FP32 conversion*/
                        drpai_output_buf1[j + size_count_hrnet] = float16_to_float32(data_ptr_hrnet[j]);
                    }
                }
                else if (InOutDataType::FLOAT32 == std::get<0>(output_buffer_hrnet))
                {
                    /*Output Data = std::get<1>(output_buffer)*/
                    float *data_ptr_hrnet = reinterpret_cast<float *>(std::get<1>(output_buffer_hrnet));
                    for (int j = 0; j < output_size_hrnet; j++)
                    {
                        drpai_output_buf1[j + size_count_hrnet] = data_ptr_hrnet[j];
                    }
                }
                else
                {
                    std::cerr << "[ERROR] Output data type : not floating point." << std::endl;
                    ret = -1;
                    break;
                }
                size_count_hrnet += output_size_hrnet;
            }

            if (ret != 0)
            {
                std::cerr << "[ERROR] DRP Inference Not working !!! " << std::endl;
                return -1;
            }

            croph[k] = det_ppl[k].bbox.h  + CROP_ADJ_X;
            cropw[k] = det_ppl[k].bbox.w  + CROP_ADJ_Y;
                         
            /*Checks that cropping height and width does not exceeds image dimension*/
            if(croph[k] < 1)
            {
                croph[k] = 1;
            }
            else if(croph[k] > IMAGE_HEIGHT)
            {
                croph[k] = IMAGE_HEIGHT;
            }
            else
            {
                /*Do Nothing*/
            }
            if(cropw[k] < 1)
            {
                cropw[k] = 1;
            }
            else if(cropw[k] > IMAGE_WIDTH)
            {
                cropw[k] = IMAGE_WIDTH;
            }
            else
            {
                /*Do Nothing*/
            }
            /*Compute Cropping Y Position based on Detection Result*/
            /*If Negative Cropping Position*/
            if(det_ppl[k].bbox.y < (croph[k]/2))
            {
                cropy[k] = 0;
            }
            else if(det_ppl[k].bbox.y > (IMAGE_HEIGHT-croph[k]/2)) /*If Exceeds Image Area*/
            {
                cropy[k] = IMAGE_HEIGHT-croph[k];
            }
            else
            {
                cropy[k] = (int16_t)det_ppl[k].bbox.y - croph[k]/2;
            }
            /*Compute Cropping X Position based on Detection Result*/
            /*If Negative Cropping Position*/
            if(det_ppl[k].bbox.x < (cropw[k]/2))
            {
                cropx[k] = 0;
            }
            else if(det_ppl[k].bbox.x > (IMAGE_WIDTH-cropw[k]/2)) /*If Exceeds Image Area*/
            {
                cropx[k] = IMAGE_WIDTH-cropw[k];
            }
            else
            {
                cropx[k] = (int16_t)det_ppl[k].bbox.x - cropw[k]/2;
            }
                            

            /*Checks that combined cropping position with width and height does not exceed the image dimension*/
            if(cropx[k] + cropw[k] > IMAGE_WIDTH)
            {
                cropw[k] = IMAGE_WIDTH - cropx[k];
            }
            if(cropy[k] + croph[k] > IMAGE_HEIGHT)
            {
                croph[k] = IMAGE_HEIGHT - cropy[k];
            }
            /*Post process start time for hrnet model*/
            R_Post_Proc_HRNet(&drpai_output_buf1[0],k);
            if(lowest_kpt_score_local[k] > 0)
            {
                R_HRNet_Coord_Convert(k);
            }
            
            /*Postprocess time end for hrnet model*/
            auto t5_hrnet = std::chrono::high_resolution_clock::now();

            auto r_post_proc_time_hrnet = std::chrono::duration_cast<std::chrono::microseconds>(t5_hrnet - t4_hrnet).count();
            auto pre_proc_time_hrnet = std::chrono::duration_cast<std::chrono::microseconds>(t1_hrnet - t0_hrnet).count();
            
            POST_PROC_TIME_HRNET_MICRO = POST_PROC_TIME_HRNET_MICRO + r_post_proc_time_hrnet;
            PRE_PROC_TIME_HRNET_MICRO = PRE_PROC_TIME_HRNET_MICRO + pre_proc_time_hrnet;
            INF_TIME_HRNET_MICRO = INF_TIME_HRNET_MICRO + inf_duration_hrnet;

        }
        POST_PROC_TIME_HRNET = POST_PROC_TIME_HRNET_MICRO/1000.0;
        PRE_PROC_TIME_HRNET = PRE_PROC_TIME_HRNET_MICRO/1000.0;
        INF_TIME_HRNET = INF_TIME_HRNET_MICRO/1000.0;
        float total_time = float(POST_PROC_TIME_HRNET_MICRO) + float(PRE_PROC_TIME_HRNET_MICRO) + float(INF_TIME_HRNET_MICRO);
        TOTAL_TIME_HRNET = total_time/1000.0;

    }
    
    
    auto r_post_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
    auto pre_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    POST_PROC_TIME_TINYYOLO = r_post_proc_time/1000.0;
    PRE_PROC_TIME_TINYYOLO = pre_proc_time/1000.0;
    INF_TIME_TINYYOLO = inf_duration/1000.0;

    float total_time_tinyyolo = float(POST_PROC_TIME_TINYYOLO) + float(PRE_PROC_TIME_TINYYOLO) + float(INF_TIME_TINYYOLO);
    TOTAL_TIME = TOTAL_TIME_HRNET + total_time_tinyyolo;
    return 0;
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
    int32_t i = 0;
    int32_t baseline = 10;
    uint8_t * img_buffer0;

    img_buffer0 = (unsigned char*) (malloc(DISP_OUTPUT_WIDTH*DISP_OUTPUT_HEIGHT*BGRA_CHANNEL));
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
            int ret = Fall_Detection();
            if (ret != 0)
            {
                std::cerr << "[ERROR] Inference Not working !!! " << std::endl;
            }
            
            draw_skeleton();

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
            
            /*HRNet model Timings*/
            stream.str("");
            stream << "HRNet";
            str = stream.str();
            Size hrnet_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - hrnet_size.width - RIGHT_ALIGN_OFFSET), (MODEL_NAME_2_Y + hrnet_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - hrnet_size.width - RIGHT_ALIGN_OFFSET), (MODEL_NAME_2_Y + hrnet_size.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);

            stream.str("");
            stream << "Pre-Proc: " << fixed << setprecision(2) << PRE_PROC_TIME_HRNET<<" ms";
            str = stream.str();
            Size pre_proc_size_hrnet = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - pre_proc_size_hrnet.width - RIGHT_ALIGN_OFFSET), (PRE_TIME_STR_Y_HRNET + pre_proc_size_hrnet.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - pre_proc_size_hrnet.width - RIGHT_ALIGN_OFFSET), (PRE_TIME_STR_Y_HRNET + pre_proc_size_hrnet.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);
            stream.str("");
            stream << "Inference: " << fixed << setprecision(2) << INF_TIME_HRNET<<" ms";
            str = stream.str();
            Size inf_size_hrnet = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - inf_size_hrnet.width - RIGHT_ALIGN_OFFSET), (I_TIME_STR_Y_HRNET + inf_size_hrnet.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - inf_size_hrnet.width - RIGHT_ALIGN_OFFSET), (I_TIME_STR_Y_HRNET + inf_size_hrnet.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);
            stream.str("");
            stream << "Post-Proc: " << fixed << setprecision(2) << POST_PROC_TIME_HRNET << " ms";
            str = stream.str();
            Size post_proc_size_hrnet = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - post_proc_size_hrnet.width - RIGHT_ALIGN_OFFSET), (P_TIME_STR_Y_HRNET + post_proc_size_hrnet.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
            putText(output_image, str,Point((DISP_OUTPUT_WIDTH - post_proc_size_hrnet.width - RIGHT_ALIGN_OFFSET), (P_TIME_STR_Y_HRNET + post_proc_size_hrnet.height)), FONT_HERSHEY_SIMPLEX, 
                        CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);

            if(person_cnt > 0)
            {
            for(i=0; i < person_cnt; i++)
            {
            if((fall_frame_count[i] >= 5 )&&(flag[i] == 1))
            {
                stream.str("");
                stream << "The person had fallen !!";
                str = stream.str();
                putText(g_frame, str, Point(PERSON_STR_X, PERSON_STR_Y), FONT_HERSHEY_SIMPLEX, PERSON_SCALE_SMALL, Scalar(0, 255, 255), PERSON_CHAR_THICKNESS );
            }
            }
            }

            Size size(DISP_INF_WIDTH, DISP_INF_HEIGHT);
            /*resize the image to the keep ratio size*/
            resize(g_frame, g_frame, size);            
            g_frame.copyTo(output_image(Rect(0, 0, DISP_INF_WIDTH, DISP_INF_HEIGHT)));
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
    /*Set Termination Request Semaphore to 0*/
    free(img_buffer0);
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
    std::cout << "Starting Elderly Fall Detection Application" << std::endl;

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
    runtime_status = runtime.LoadModel(model_dir, drpaimem_addr_start + DRPAI_MEM_OFFSET);
        if(!runtime_status)
    {
        std::cerr << "[ERROR] Failed to load model. " << std::endl;
        close(drpai_fd);
        return -1;
    }    
 
    std::cout << "[INFO] loaded runtime model :" << model_dir << "\n\n";
    /*Load model_dir structure and its weight to runtime object */
    runtime_status1 = runtime1.LoadModel(model_dir1, drpaimem_addr_start + DRPAI_MEM_OFFSET1);
    
    if(!runtime_status1)
    {
        std::cerr << "[ERROR] Failed to load model. " << std::endl;
        close(drpai_fd);
        return -1;
    }    

    std::cout << "[INFO] loaded runtime model :" << model_dir1 << "\n\n";

    switch (input_source_map[input_source])
    {
        /* Input Source : USB*/
        case 1:{
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
