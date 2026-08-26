cmake_minimum_required (VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)
project(10_Driver_Monitoring_System)

if(NOT DEFINED ENV{TVM_ROOT})
  message(SEND_ERROR "Need to specify location of drp-tvm with option TVM_ROOT")
endif()

if(NOT DEFINED ENV{PRODUCT})
  message(SEND_ERROR "Need to specify product name with option PRODUCT")
endif()

# Setting and including TVM Directory
set(TVM_ROOT "$ENV{TVM_ROOT}")
include_directories(${TVM_ROOT}/tvm/include)
include_directories(${TVM_ROOT}/setup/include)
include_directories(${TVM_ROOT}/tvm/3rdparty/dlpack/include)
include_directories(${TVM_ROOT}/tvm/3rdparty/dmlc-core/include)
include_directories(${TVM_ROOT}/tvm/3rdparty/compiler-rt)
include_directories(./)
include_directories(${TVM_ROOT}/apps)
include_directories(${TVM_ROOT}/apps/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${TVM_ROOT}/3rdparty/asio/asio/include)
include_directories(${TVM_ROOT}/3rdparty/spdlog/include)

set(LIBMERA_RT_PATH ${TVM_ROOT}/obj/build_runtime/v2h/lib)
add_definitions(-DMERA_DRP_RUNTIME)
set(EXE_NAME "dms_detection_app")
set(PRODUCT $ENV{PRODUCT})

if(PRODUCT STREQUAL "V2H")
    include_directories(${CMAKE_SOURCE_DIR}/ruhmi_2026-06)
    file(GLOB SOURCE *.cpp *.h ${CMAKE_SOURCE_DIR}/ruhmi_2026-06/*.cpp ${CMAKE_SOURCE_DIR}/ruhmi_2026-06/*.h)
    include_directories(${CMAKE_SYSROOT}/usr/src/debug/gstreamer1.0-plugins-base/1.22.12/gst-libs/gst/gl)
    list(REMOVE_ITEM SOURCE ${CMAKE_SOURCE_DIR}/ruhmi_2026-06/PreRuntime.cpp)
    list(REMOVE_ITEM SOURCE ${CMAKE_SOURCE_DIR}/wayland.cpp)
    list(APPEND SOURCE ${CMAKE_SYSROOT}/usr/src/debug/gstreamer1.0-plugins-base/1.22.12/gst-libs/gst/gl/xdg-shell-client-protocol.c)
    add_definitions(-DV2H)
    message("-- PRODUCT [V2H]")
else()
    include_directories(${CMAKE_SOURCE_DIR}/ruhmi_2026-06)
    file(GLOB SOURCE *.cpp *.h ${CMAKE_SOURCE_DIR}/ruhmi_2026-06/*.cpp ${CMAKE_SOURCE_DIR}/ruhmi_2026-06/*.h)
    include_directories(${CMAKE_SYSROOT}/usr/src/debug/gstreamer1.0-plugins-base/1.22.12/gst-libs/gst/gl)
    list(REMOVE_ITEM SOURCE ${CMAKE_SOURCE_DIR}/ruhmi_2026-06/PreRuntime.cpp)
    list(REMOVE_ITEM SOURCE ${CMAKE_SOURCE_DIR}/wayland.cpp)
    list(APPEND SOURCE ${CMAKE_SYSROOT}/usr/src/debug/gstreamer1.0-plugins-base/1.22.12/gst-libs/gst/gl/xdg-shell-client-protocol.c)
    add_definitions(-DV2H -DV2N)
    message("-- PRODUCT [V2N]")
endif()
add_executable(${EXE_NAME} ${SOURCE})
target_compile_options(${EXE_NAME} PRIVATE -O3 -mtune=cortex-a55 -Wall)


set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,${LIBMERA_RT_PATH} -Wl,-rpath-link,${LIBMERA_RT_PATH}")


target_compile_features(${EXE_NAME} PUBLIC cxx_std_17)


target_link_directories(${EXE_NAME} PRIVATE ${LIBMERA_RT_PATH})

target_link_libraries(${EXE_NAME}
	mera2_runtime
	mera2_plan_io
	drp_tvm_rt
	pthread glib-2.0 mmngr gobject-2.0 gstreamer-1.0  gstbase-1.0 gstapp-1.0
	opencv_imgcodecs opencv_imgproc opencv_core opencv_highgui
	turbojpeg jpeg webp tiff z tbb gtk-3 png16 gdk-3 cairo
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

if(PRODUCT STREQUAL "V2H")
    target_compile_definitions(${EXE_NAME} PRIVATE V2H)
endif()
if(PRODUCT STREQUAL "V2N")
    target_compile_definitions(${EXE_NAME} PRIVATE V2N)
endif()