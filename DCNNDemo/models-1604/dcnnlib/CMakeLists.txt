cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project("dcnnlib")

find_package(OpenCV REQUIRED)

add_library(dcnnlib SHARED dcnnmodule.cpp)
target_link_libraries(dcnnlib ${OpenCV_LIBS})
include_directories(dcnnlib /usr/include/python3.5m)
target_link_libraries(dcnnlib /usr/lib/x86_64-linux-gnu/libpython3.5m.so)
include_directories(dcnnlib pyboostcvconverter/include/pyboostcvconverter)
link_libraries(dcnnlib pyboostcvconverter/build)

set_property(TARGET dcnnlib PROPERTY CXX_STANDARD 11)
