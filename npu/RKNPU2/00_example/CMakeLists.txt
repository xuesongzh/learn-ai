# CMake需要的最低版本号是3.4.1
cmake_minimum_required(VERSION 3.4.1)
# 指定了项目的名称为resnet18
project(example)

# 不使用任何C编译器标志,使用C++11标准进行编译
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../3rdparty/librknn_api)  #该变量定义了RKNN API库的路径
set(LIB_ARCH aarch64)
set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)  # 该变量定义了RKNN运行时库的路径。
include_directories(${RKNN_API_PATH}/include)  # 添加RKNN API的头文件

# opencv
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/opencv/share/OpenCV)  # 指定了OpenCV库的路径。
find_package(OpenCV REQUIRED)  # 自动查找OpenCV库，并将相关信息保存在CMake内置变量中

# 指定了程序运行时查找动态链接库的路径
set(CMAKE_INSTALL_RPATH "lib")

# 创建一个可执行文件
add_executable(${PROJECT_NAME}
    src/main.cc
)

# 指定工程需要链接的库文件
target_link_libraries(${PROJECT_NAME}
  ${RKNN_RT_LIB}
  ${OpenCV_LIBS}
)

# 指定了程序安装的路径
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/${PROJECT_NAME}_${CMAKE_SYSTEM_NAME})

# 拷贝可执行程序和需要用的库以及后面测试要用到的model测试文件
install(TARGETS ${PROJECT_NAME} DESTINATION ./)
install(DIRECTORY model DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
