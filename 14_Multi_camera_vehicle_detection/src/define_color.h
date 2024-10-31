/***********************************************************************************************************************
* Copyright (C) 2023 Renesas Electronics Corporation. All rights reserved.
***********************************************************************************************************************/
/***********************************************************************************************************************
* File Name    : define_color.h
* Version      : 1.00
* Description  : RZ/V2H DRP-AI Sample Application for Multi-camera streaming detection with MIPI/USB Camera
***********************************************************************************************************************/
#ifndef DEFINE_COLOR_H
#define DEFINE_COLOR_H

/*****************************************
* color
******************************************/
/* Pascal VOC dataset label list */
const static std::vector<std::string> label_file_map = 
{ 
    "ambulance",
    "bicycle",
    "bike",
    "car",
    "fire engine",
    "police car",
    "truck",
    "auto",
    "bus",
    "tractor"
};

/* box color list */
const static unsigned int box_color[] =
{
    (0xFFFF00u),
    (0xFF0000u),
    (0xC0C0C0u),
    (0xFFA07Au),
    (0xFF1493u),
    (0x006400u),
    (0x00BFFFu),
    (0xDAA520u),
    (0xFF00FFu),
    (0xFFC0CBu)
};


#endif

