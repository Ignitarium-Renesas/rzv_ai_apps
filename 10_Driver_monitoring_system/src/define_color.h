/***********************************************************************************************************************
* Copyright (C) 2023 Renesas Electronics Corporation. All rights reserved.
***********************************************************************************************************************/
/***********************************************************************************************************************
* File Name    : define_color.h
* Version      : 1.00
* Description  : RZ/V2H DRP-AI Sample Application for Driver monitoring system detection with MIPI/USB Camera
***********************************************************************************************************************/
#ifndef DEFINE_COLOR_H
#define DEFINE_COLOR_H

/*****************************************
* color
******************************************/
/* Pascal VOC dataset label list */
#ifdef V2N

const static std::vector<std::string> label_file_map =
{
    "eyes_closed",
    "eyes_open",
    "center",
    "down",
    "left",
    "right",
    "no_yawn",
    "yawning"
};

#else

const static std::vector<std::string> label_file_map =
{
    "eyes_closed",
    "eyes_open",
    "center",
    "down",
    "left",
    "right",
    "no_yawn",
    "seatbelt",
    "using_mobile",
    "yawning"
};

#endif

#endif // DEFINE_COLOR_H