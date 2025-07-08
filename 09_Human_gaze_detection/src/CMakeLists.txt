cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
project(09_Human_gaze_detection_app)

# Setting and including TVM Directory
set(TVM_ROOT $ENV{TVM_HOME})
include_directories(${TVM_ROOT}/include)
include_directories(${TVM_ROOT}/3rdparty/dlpack/include)
include_directories(${TVM_ROOT}/3rdparty/dmlc-core/include)
include_directories(${TVM_ROOT}/3rdparty/compiler-rt)

set(TVM_RUNTIME_LIB ${TVM_ROOT}/build_runtime/libtvm_runtime.so)
set(EXE_NAME "human_gaze_detection_app")
set(PRODUCT $ENV{PRODUCT})

if(PRODUCT STREQUAL "V2H")
    include_directories(${CMAKE_SOURCE_DIR}/drp-ai_tvm_v230)
    file(GLOB SOURCE *.cpp *.h ${CMAKE_SOURCE_DIR}/drp-ai_tvm_v230/*.cpp ${CMAKE_SOURCE_DIR}/drp-ai_tvm_v230/*.h)
    list(REMOVE_ITEM SOURCE ${CMAKE_SOURCE_DIR}/drp-ai_tvm_v230/PreRuntime.cpp)
    list(REMOVE_ITEM SOURCE ${CMAKE_SOURCE_DIR}/wayland_scarthgap.cpp)
    add_definitions(-DV2H)
    message("-- PRODUCT [V2H]") 
elseif(PRODUCT STREQUAL "V2N")
    include_directories(${CMAKE_SOURCE_DIR}/drp-ai_tvm_v251)
    file(GLOB SOURCE *.cpp *.h ${CMAKE_SOURCE_DIR}/drp-ai_tvm_v251/*.cpp ${CMAKE_SOURCE_DIR}/drp-ai_tvm_v251/*.h)
    include_directories(${CMAKE_SYSROOT}/usr/src/debug/gstreamer1.0-plugins-base/1.22.12/gst-libs/gst/gl)
    list(REMOVE_ITEM SOURCE ${CMAKE_SOURCE_DIR}/drp-ai_tvm_v251/PreRuntime.cpp)
    list(REMOVE_ITEM SOURCE ${CMAKE_SOURCE_DIR}/wayland.cpp)
    list(APPEND SOURCE ${CMAKE_SYSROOT}/usr/src/debug/gstreamer1.0-plugins-base/1.22.12/gst-libs/gst/gl/xdg-shell-client-protocol.c)
    add_definitions(-DV2H -DV2N)
    message("-- PRODUCT [V2N]")
else()
    include_directories(${CMAKE_SOURCE_DIR}/drp-ai_tvm_v230)
    file(GLOB SOURCE *.cpp *.h ${CMAKE_SOURCE_DIR}/drp-ai_tvm_v230/*.cpp ${CMAKE_SOURCE_DIR}/drp-ai_tvm_v230/*.h)
    list(REMOVE_ITEM SOURCE ${CMAKE_SOURCE_DIR}/drp-ai_tvm_v230/PreRuntimeV2H.cpp)
    list(REMOVE_ITEM SOURCE ${CMAKE_SOURCE_DIR}/wayland_scarthgap.cpp)
    add_definitions(-DV2L)
    message("-- PRODUCT [V2L]")
endif()
add_executable(${EXE_NAME} ${SOURCE})

TARGET_LINK_LIBRARIES(${EXE_NAME} pthread glib-2.0 mmngr gobject-2.0 gstreamer-1.0  gstbase-1.0 gstapp-1.0 
	opencv_imgcodecs opencv_imgproc opencv_core opencv_highgui
	jpeg webp tiff z tbb gtk-3 png16 gdk-3 cairo
	lzma rt cairo-gobject
	xkbcommon wayland-cursor wayland-egl wayland-client epoxy
	fribidi harfbuzz fontconfig
	glib-2.0 gobject-2.0 gdk_pixbuf-2.0 gmodule-2.0 pangocairo-1.0
	atk-1.0 gio-2.0 pango-1.0 freetype pixman-1 uuid pcre
	mount resolv expat pangoft2-1.0 blkid
	EGL GLESv2 mmngrbuf)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()
target_link_libraries(${EXE_NAME} ${OpenCV_LIBS})
target_link_libraries(${EXE_NAME} ${TVM_RUNTIME_LIB})
if(PRODUCT STREQUAL "V2H")
    target_compile_definitions(${EXE_NAME} PRIVATE V2H)
endif()
if(PRODUCT STREQUAL "V2N")
    target_compile_definitions(${EXE_NAME} PRIVATE V2N)
endif()
