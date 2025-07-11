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
* Copyright 2024 Renesas Electronics Corporation
* 
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
* 
* http://www.apache.org/licenses/LICENSE-2.0
* 
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
***********************************************************************************************************************/
/***********************************************************************************************************************
* File Name    : wayland.cpp
* Version      : v5.00
* Description  : RZ/V AI SDK Sample Application for Object Detection
***********************************************************************************************************************/

/*****************************************
 * Includes
 ******************************************/
#include "define.h"
#include "wayland.h"
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <fstream>


struct WaylandGlobals {
    struct wl_compositor* compositor;
    struct xdg_wm_base* wm_base;
};

/*****************************************
 * Function Name : xdg_wm_base_ping
 * Description   : xdg_wm_base_listener callback
 *                 Notifies the compositor that the client is responsive.
 * Arguments     : data    = User data passed when adding the listener.
 *                 wm_base = The xdg_wm_base interface that received the ping event.
 *                 serial  = Identification ID is notified.
 * Return value  : -
 ******************************************/
static void xdg_wm_base_ping(void* data, 
                            struct xdg_wm_base* wm_base, 
                            uint32_t serial) 
{
    xdg_wm_base_pong(wm_base, serial);
}

static const struct xdg_wm_base_listener xdg_wm_base_listener = {
    .ping = xdg_wm_base_ping,
};

/*****************************************
 * Function Name : xdg_surface_configure
 * Description   : xdg_surface_listener callback  
 *                 Acknowledges a configure event from the compositor.
 * Arguments     : data    = User data passed when adding the listener.
 *                 surface = The xdg_surface interface that received the configure event.
 *                 serial  = Identification ID is notified.
 * Return value  : -
 ******************************************/
static void xdg_surface_configure(void* data, struct xdg_surface* surface, uint32_t serial) {
    xdg_surface_ack_configure(surface, serial);
}

static const struct xdg_surface_listener xdg_surface_listener = {
    .configure = xdg_surface_configure,
};

Wayland::Wayland()
{
}

Wayland::~Wayland()
{
}

/*****************************************
 * Function Name : registry_global
 * Description   : wl_registry_listener callback
 *                 wayland func bind.
 * Arguments     : data      = The third argument of wl_registry_add_listener() is notified.
 *                 regisry   = The first argument of wl_registry_add_listener() is notified.
 *                 name      = global object ID is notified.
 *                 interface = interface name is notifed.
 *                 version   = interface version is notified.
 * Return value  : -
 ******************************************/
static void registry_global(void *data,
                            struct wl_registry *registry, uint32_t id,
                            const char *interface, uint32_t version)
{
    struct WaylandGlobals* globals = (struct WaylandGlobals*)data;
    if (strcmp(interface, "wl_compositor") == 0) {
        globals->compositor = (struct wl_compositor*)wl_registry_bind(registry, id, &wl_compositor_interface, 1);
    }
    else if (strcmp(interface, "xdg_wm_base") == 0) {
        globals->wm_base = (struct xdg_wm_base*)wl_registry_bind(registry, id, &xdg_wm_base_interface, 1);
        xdg_wm_base_add_listener(globals->wm_base, &xdg_wm_base_listener, NULL);
    }
}

/* registry callback for listener */
static const struct wl_registry_listener registry_listener = { registry_global, NULL };

/*****************************************
 * Function Name : LoadShader
 * Description   : Return the loaded and compiled shader
 * Arguments     : type
 *                 shaderSrc
 * Return value  : 0 if Success
 *                 not 0 otherwise
 ******************************************/
GLuint Wayland::LoadShader(GLenum type, const char* shaderSrc)
{
    GLuint shader = glCreateShader(type);
    assert(shader);

    glShaderSource(shader, 1, &shaderSrc, NULL);
    glCompileShader(shader);

    GLint compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    assert(compiled);

    return shader;
}

/*****************************************
 * Function Name : initProgramObject
 * Description   : Initialize the shaders and return the program object
 * Arguments     : pShader
 * Return value  : 0 if Success
 *                 not 0 otherwise
 ******************************************/
GLuint Wayland::initProgramObject(SShader* pShader)
{
    const char* vshader = R"(
        attribute vec4 position;
        attribute vec2 texcoord;
        varying vec2 texcoordVarying;
        void main() {
            gl_Position = position;
            texcoordVarying = texcoord;
        }
    )";

    const char* fshader = R"(
        precision mediump float;
        uniform sampler2D texture;
        varying vec2 texcoordVarying;
        void main() {
            highp float r = texture2D(texture, texcoordVarying).b;
            highp float g = texture2D(texture, texcoordVarying).g;
            highp float b = texture2D(texture, texcoordVarying).r;
            highp float a = texture2D(texture, texcoordVarying).a;
            gl_FragColor = vec4(r,g,b,a);
        }

    )";

    GLuint vertexShader = LoadShader(GL_VERTEX_SHADER, vshader);
    GLuint fragmentShader = LoadShader(GL_FRAGMENT_SHADER, fshader);

    GLuint programObject = glCreateProgram();
    assert(programObject);

    glAttachShader(programObject, vertexShader);
    glAttachShader(programObject, fragmentShader);

    glLinkProgram(programObject);

    GLint linked;
    glGetProgramiv(programObject, GL_LINK_STATUS, &linked);
    assert(linked);

    glDeleteShader(fragmentShader);
    glDeleteShader(vertexShader);

    pShader->unProgram = programObject;
    pShader->nAttrPos = glGetAttribLocation(pShader->unProgram, "position");
    pShader->nAttrColor = glGetAttribLocation(pShader->unProgram, "texcoord");
    return programObject;
}

/*****************************************
 * Function Name : initEGLDisplay
 * Description   : Configure EGL and return necessary resources
 * Arguments     : nativeDisplay
 *                 nativeWindow
 *                 eglDisplay
 *                 eglSurface
 * Return value  : 0 if Success
 *                 not 0 otherwise
 ******************************************/
static int8_t initEGLDisplay(EGLNativeDisplayType nativeDisplay, EGLNativeWindowType nativeWindow, EGLDisplay* eglDisplay, EGLSurface* eglSurface)
{
    EGLint number_of_config;
    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_NONE
    };

    static const EGLint context_attribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL_NONE
    };

    *eglDisplay = eglGetDisplay(nativeDisplay);
    if (*eglDisplay == EGL_NO_DISPLAY)
    {
        return -1;
    }

    EGLBoolean initialized = eglInitialize(*eglDisplay, NULL, NULL);
    if (initialized != EGL_TRUE)
    {
        return -1;
    }

    EGLConfig configs[1];

    EGLBoolean config = eglChooseConfig(*eglDisplay, config_attribs, configs, 1, &number_of_config);
    if (config != EGL_TRUE)
    {
        return -1;
    }

    EGLContext eglContext = eglCreateContext(*eglDisplay, configs[0], EGL_NO_CONTEXT, context_attribs);

    *eglSurface = eglCreateWindowSurface(*eglDisplay, configs[0], nativeWindow, NULL);
    if (*eglSurface == EGL_NO_SURFACE)
    {
        return -1;
    }

    EGLBoolean makeCurrent = eglMakeCurrent(*eglDisplay, *eglSurface, *eglSurface, eglContext);
    if (makeCurrent != EGL_TRUE)
    {
        return -1;
    }
    return 0;
}


/*****************************************
 * Function Name : initWaylandDisplay
 * Description   : Connect to the Wayland display and return the display and the surface
 * Arguments     : wlDisplay
 *                 wlSurface
 * Return value  : 0 if Success
 *                 not 0 otherwise
 ******************************************/
static int8_t initWaylandDisplay(struct wl_display** wlDisplay, struct wl_surface** wlSurface)
{
    struct WaylandGlobals globals = { 0 };

    *wlDisplay = wl_display_connect(NULL);
    if(*wlDisplay == NULL)
    {
        return -1;
    }

    struct wl_registry* registry = wl_display_get_registry(*wlDisplay);
    wl_registry_add_listener(registry, &registry_listener, (void*)&globals);

    wl_display_dispatch(*wlDisplay);
    wl_display_roundtrip(*wlDisplay);
    if (globals.compositor == NULL || globals.wm_base == NULL)
    {
        return -1;
    }

    *wlSurface = wl_compositor_create_surface(globals.compositor);
    if (*wlSurface == NULL)
    {
        return -1;
    }

    struct xdg_surface* xdg_surface = xdg_wm_base_get_xdg_surface(globals.wm_base, *wlSurface);
    if (xdg_surface == NULL)
    {
        return -1;
    }

    struct xdg_toplevel* xdg_toplevel = xdg_surface_get_toplevel(xdg_surface);
    if (xdg_toplevel == NULL) {
        return -1;
    }

    xdg_surface_add_listener(xdg_surface, &xdg_surface_listener, NULL);

    wl_surface_commit(*wlSurface);
    return 0;
}

/*****************************************
 * Function Name : initWindow
 * Description   : Connect Wayland and make EGL
 * Arguments     : width
 *                 height
 *                 wlDisplay
 *                 eglDisplay
 *                 eglSurface
 * Return value  : 0 if Success
 *                 not 0 otherwise
 ******************************************/
static int8_t initWindow(GLint width, GLint height, struct wl_display** wlDisplay, EGLDisplay* eglDisplay, EGLSurface* eglSurface)
{
    int8_t ret = 0;
    struct wl_surface* wlSurface;
    ret = initWaylandDisplay(wlDisplay, &wlSurface);
    if (ret != 0)
    {
        return -1;
    }

    struct wl_egl_window* wlEglWindow = wl_egl_window_create(wlSurface, width, height);
    if (wlEglWindow == NULL)
    {
        return -1;
    }

    ret = initEGLDisplay((EGLNativeDisplayType)*wlDisplay, (EGLNativeWindowType)wlEglWindow, eglDisplay, eglSurface);
    if (ret != 0)
    {
        return -1;
    }
    return 0;
}

/*****************************************
 * Function Name : init
 * Description   : wayland client init
 *                 create buffer.
 * Arguments     : w  = width
 *                 h  = height
 *                 c  = color channel
 *                 overlay  = flag for alpha blending
 * Return value  : 0 if Success
 *                 not 0 otherwise
 ******************************************/
uint8_t Wayland::init(uint32_t w, uint32_t h, uint32_t c, bool overlay)
{
    int8_t ret = 0;
    img_w = w;
    img_h = h;
    img_c = c;
    img_overlay = overlay;

    // Connect Wayland and make EGL
    ret = initWindow(w, h, &display, &eglDisplay, &eglSurface);
    if (ret != 0)
    {
        return -1;
    }

    //Initialize the shaders and return the program object
    GLuint programObject = initProgramObject(&sShader);
    if (programObject == 0)
    {
        return -1;
    }

    // Apply program object
    glUseProgram(sShader.unProgram);
    glGenTextures(2, textures);

    glEnableVertexAttribArray(sShader.nAttrPos);
    glEnableVertexAttribArray(sShader.nAttrColor);

    // enable Alpha Blending
    if (img_overlay == true){
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    glUniform1i(glGetUniformLocation(sShader.unProgram, "texture"), 0);

    return 0;
}

/*****************************************
 * Function Name : exit
 * Description   : Exit Wayland
 * Arguments     : -
 * Return value  : 0 if Success
 *                 not 0 otherwise
 ******************************************/
uint8_t Wayland::exit()
{
    SShader* pShader = &sShader;
    if (pShader) {
        glDeleteProgram(pShader->unProgram);
        pShader->unProgram = 0;
        pShader->nAttrPos = -1;
        pShader->nAttrColor = -1;
    }

    if (xdg_toplevel) {
        xdg_toplevel_destroy(xdg_toplevel);
        xdg_toplevel = NULL;
    }

    if (xdg_surface) {
        xdg_surface_destroy(xdg_surface);
        xdg_surface = NULL;
    }

    if (wm_base) {
        xdg_wm_base_destroy(wm_base);
        wm_base = NULL;
    }

    if (display) {
        wl_display_disconnect(display);
        display = NULL;
    }
    return 0;
}


/*****************************************
 * Function Name : render
 * Description   : 
 * Arguments     : pShader
 *                 texID
 * Return value  : 0 if Success
 *                 not 0 otherwise
 ******************************************/
uint8_t Wayland::render(SShader* pShader, GLuint texID)
{
    const float vertices[] = {
        -1.0f,  1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
         1.0f, -1.0f, 0.0f
    };

    const float texcoords[] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f };


    glVertexAttribPointer(pShader->nAttrColor, 2, GL_FLOAT, GL_FALSE, 0, texcoords);
    glVertexAttribPointer(pShader->nAttrPos, 3, GL_FLOAT, GL_FALSE, 0, vertices);

    // draw texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texID);
    //glUniform1i(uniID, texID);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    return 0;
}


/*****************************************
 * Function Name : setupTexture
 * Description   : Bind Texture
 * Arguments     : texID
 *                 src_pixels
 * Return value  : 0 if Success
 *                 not 0 otherwise
 ******************************************/
uint8_t Wayland::setupTexture(GLuint texID, uint8_t* src_pixels)
{
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, texID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_w, img_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, src_pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return 0;
}


/*****************************************
 * Function Name : commit
 * Description   : Commit to update the display image
 * Arguments     : buf_id = buffer id
 * Return value  : 0 if Success
 *                 not 0 otherwise
 ******************************************/
uint8_t Wayland::commit(uint8_t* cam_buffer, uint8_t* ol_buffer)
{
    uint8_t ret = 0;
#ifdef DEBUG_TIME_FLG
    using namespace std;
    chrono::system_clock::time_point start, end;
    double time = 0;
    start = chrono::system_clock::now();
#endif // DEBUG_TIME_FLG

    // setup texture
    setupTexture(textures[0], cam_buffer);
    if (ol_buffer != NULL && img_overlay == true) {
        setupTexture(textures[1], ol_buffer);
    }
#ifdef DEBUG_TIME_FLG
    end = chrono::system_clock::now();
    time = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0);
    printf("Setup Image Time          : %lf[ms]\n", time);
#endif // DEBUG_TIME_FLG

    // clear
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#ifdef DEBUG_TIME_FLG
    start = chrono::system_clock::now();
#endif // DEBUG_TIME_FLG

    // render
    render(&sShader, textures[0]);
    if (ol_buffer != NULL && img_overlay == true) {
        render(&sShader, textures[1]);
    }
#ifdef DEBUG_TIME_FLG
    end = chrono::system_clock::now();
    time = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0);
    printf("Specifies Render Time     : %lf[ms]\n", time);
    start = chrono::system_clock::now();
#endif // DEBUG_TIME_FLG

    eglSwapBuffers(eglDisplay, eglSurface);

#ifdef DEBUG_TIME_FLG
    end = chrono::system_clock::now();
    time = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0);
    printf("Update Frame Time         : %lf[ms]\n", time);
#endif // DEBUG_TIME_FLG

    return ret;
}

