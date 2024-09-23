/***********************************************************************************************************************
* DISCLAIMER
* This software is supplied by Renesas Electronics Corporation and is only intended for use with Renesas products. No
* other uses are authorized. This software is owned by Renesas Electronics Corporation and is protected under all
* applicable laws, including copyright laws.
* THIS SOFTWARE IS PROVIDED "AS IS" AND RENESAS MAKES NO WARRANTIES REGARDING
* THIS SOFTWARE, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. ALL SUCH WARRANTIES ARE EXPRESSLY DISCLAIMED. TO THE MAXIMUM
* EXTENT PERMITTED NOT PROHIBITED BY LAW, NEITHER RENESAS ELECTRONICS CORPORATION NOR ANY OF ITS AFFILIATED COMPANIES
* SHALL BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES FOR ANY REASON RELATED TO THIS
* SOFTWARE, EVEN IF RENESAS OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
* Renesas reserves the right, without notice, to make changes to this software and to discontinue the availability of
* this software. By using this software, you agree to the additional terms and conditions found by accessing the
* following link:
* http://www.renesas.com/disclaimer
*
* Copyright (C) 2024 Renesas Electronics Corporation. All rights reserved.
***********************************************************************************************************************/
/***********************************************************************************************************************
* File Name    : dmabuf.h
* Version      : v1.00
* Description  : RZ/V2H Multi Camera Vehicle Detection Application
***********************************************************************************************************************/

#ifndef DMABUF_H
#define DMABUF_H

#include "define.h"

/* This block of code is only accessible from C code. */
#ifdef __cplusplus
extern "C" {
#endif
#include "mmngr_user_public.h"
#include "mmngr_buf_user_public.h"
#ifdef __cplusplus
}
#endif

/*****************************************
* dma_buffer : dma buffer itself and its feature
******************************************/
typedef struct 
{
        /* The index of the buffer. */
        uint32_t idx;
        /* The file descriptor for the DMA buffer. */
        uint32_t dbuf_fd;
        /* The size of the buffer in bytes. */
        uint32_t size;
        /* The physical address of DMA buffer. */
        uint32_t phy_addr;
        /* The pointer to the memory for the buffer. */
        void *mem;           
}dma_buffer;

typedef struct
{
        /* The index of the buffer. */
        uint32_t idx;
        /* The file descriptor for the DMA buffer. */
        uint32_t dbuf_fd;
        /* The size of the buffer in bytes. */
        uint32_t size;
        /* The physical address of DMA buffer. */
        uint32_t phy_addr;
        /* The pointer to the memory for the buffer. */
        void *mem;
}dma_buffer1;

typedef struct
{
        /* The index of the buffer. */
        uint32_t idx;
        /* The file descriptor for the DMA buffer. */
        uint32_t dbuf_fd;
        /* The size of the buffer in bytes. */
        uint32_t size;
        /* The physical address of DMA buffer. */
        uint32_t phy_addr;
        /* The pointer to the memory for the buffer. */
        void *mem;
}dma_buffer2;

typedef struct
{
        /* The index of the buffer. */
        uint32_t idx;
        /* The file descriptor for the DMA buffer. */
        uint32_t dbuf_fd;
        /* The size of the buffer in bytes. */
        uint32_t size;
        /* The physical address of DMA buffer. */
        uint32_t phy_addr;
        /* The pointer to the memory for the buffer. */
        void *mem;
}dma_buffer3;

/*****************************************
* Functions
******************************************/
int8_t buffer_alloc_dmabuf(dma_buffer *buffer,int buf_size);
int8_t buffer_alloc_dmabuf1(dma_buffer1 *buffer1,int buf_size);
int8_t buffer_alloc_dmabuf2(dma_buffer2 *buffer2,int buf_size);
int8_t buffer_alloc_dmabuf3(dma_buffer3 *buffer3,int buf_size);
int buffer_flush_dmabuf(uint32_t idx, uint32_t size);
void buffer_free_dmabuf(dma_buffer *buffer);

#endif //DMABUF_H