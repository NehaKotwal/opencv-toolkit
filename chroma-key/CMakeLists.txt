cmake_minimum_required(VERSION 3.15)
set (CMAKE_CXX_STANDARD 17)
project(chroma_key)
set(SOURCE chroma_key.cpp)
INCLUDE_DIRECTORIES(/usr/local/include/opencv4)
LINK_DIRECTORIES(/usr/local/lib)
add_executable(${PROJECT_NAME} ${SOURCE})
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
    opencv_core
    opencv_highgui
    opencv_imgcodecs
    opencv_imgproc
)
