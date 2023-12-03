cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov5_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# rknn api
set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/include/librknnrt.so)
include_directories(${CMAKE_SOURCE_DIR}/include/3rdparty)

# opencv
set(OpenCV_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/opencv)
set(OpenCV_LIBS 
  ${OpenCV_PATH}/lib/libopencv_core.so
  ${OpenCV_PATH}/lib/libopencv_highgui.so
  ${OpenCV_PATH}/lib/libopencv_imgproc.so
  ${OpenCV_PATH}/lib/libopencv_videoio.so
  ${OpenCV_PATH}/lib/libopencv_video.so
  ${OpenCV_PATH}/lib/libopencv_imgcodecs.so
  ${OpenCV_PATH}/lib/libopencv_dnn.so
  ${OpenCV_PATH}/lib/libopencv_calib3d.so
  ${OpenCV_PATH}/lib/libopencv_features2d.so
  ${OpenCV_PATH}/lib/libopencv_flann.so
  ${OpenCV_PATH}/lib/libopencv_core.so.407
  ${OpenCV_PATH}/lib/libopencv_highgui.so.407
  ${OpenCV_PATH}/lib/libopencv_imgproc.so.407
  ${OpenCV_PATH}/lib/libopencv_videoio.so.407
  ${OpenCV_PATH}/lib/libopencv_video.so.407
  ${OpenCV_PATH}/lib/libopencv_imgcodecs.so.407
  ${OpenCV_PATH}/lib/libopencv_dnn.so.407
  ${OpenCV_PATH}/lib/libopencv_calib3d.so.407
  ${OpenCV_PATH}/lib/libopencv_features2d.so.407
  ${OpenCV_PATH}/lib/libopencv_flann.so.407
  ${OpenCV_PATH}/lib/libopencv_core.so.4.7.0
  ${OpenCV_PATH}/lib/libopencv_highgui.so.4.7.0
  ${OpenCV_PATH}/lib/libopencv_imgproc.so.4.7.0
  ${OpenCV_PATH}/lib/libopencv_videoio.so.4.7.0
  ${OpenCV_PATH}/lib/libopencv_video.so.4.7.0
  ${OpenCV_PATH}/lib/libopencv_imgcodecs.so.4.7.0
  ${OpenCV_PATH}/lib/libopencv_dnn.so.4.7.0
  ${OpenCV_PATH}/lib/libopencv_calib3d.so.4.7.0
  ${OpenCV_PATH}/lib/libopencv_features2d.so.4.7.0
  ${OpenCV_PATH}/lib/libopencv_flann.so.4.7.0
)
include_directories( ${OpenCV_PATH}/include/opencv4)

#rga
set(RGA_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/rga/RK3588)
set(LIB_ARCH aarch64)
set(RGA_LIB ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so)
include_directories( ${RGA_PATH}/include)
set(CMAKE_INSTALL_RPATH "lib")

# rknn_yolov5_demo
include_directories( ${CMAKE_SOURCE_DIR}/include)

add_executable(rknn_yolov5_demo
        src/main.cc
        src/postprocess.cc
        src/Hungarian.cc  
        src/KalmanTracker.cc    
        src/sort.cc
)

target_link_libraries(rknn_yolov5_demo
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS}
)


# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo_${CMAKE_SYSTEM_NAME})
install(TARGETS rknn_yolov5_demo DESTINATION ./)
install(PROGRAMS ${OpenCV_LIBS} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(DIRECTORY model DESTINATION ./)
