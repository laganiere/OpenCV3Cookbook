# cmake for OpenCV2 Cookbook chapter 02
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# add executable
add_executable( saltImage saltImage.cpp)
add_executable( colorReduce colorReduce.cpp)

# link libraries
target_link_libraries( saltImage ${OpenCV_LIBS})
target_link_libraries( colorReduce ${OpenCV_LIBS})

FILE(COPY ${CMAKE_SOURCE_DIR}/images/boldt.jpg DESTINATION .)
