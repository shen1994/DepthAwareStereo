cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dcnndemo)

find_package(OpenCV REQUIRED)

add_executable(dcnndemo src/demo.cpp)

target_link_libraries(dcnndemo ${OpenCV_LIBS})
include_directories(dcnndemo /usr/include/python3.5m)
target_link_libraries(dcnndemo /usr/lib/x86_64-linux-gnu/libpython3.5m.so)
include_directories(dcnndemo ./models/dcnnlib/pyboostcvconverter/include/pyboostcvconverter)
link_libraries(dcnndemo ./models/dcnnlib/pyboostcvconverter/build)
include_directories(dcnndemo ./models/dcnnlib)

target_link_libraries(dcnndemo /home/westwell/Desktop/DCNNDemo/models/dcnnlib/build/libdcnnlib.so)

set_property(TARGET dcnndemo PROPERTY CXX_STANDARD 11)

install(TARGETS dcnndemo RUNTIME DESTINATION bin)
