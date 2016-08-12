# cmake for OpenCV Cookbook 3rd edition chapter 11
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# add executable
add_executable( calibrate calibrate.cpp CameraCalibrator.cpp)
add_executable( cameraPose cameraPose.cpp)

# link libraries
target_link_libraries( calibrate ${OpenCV_LIBS})
target_link_libraries( cameraPose ${OpenCV_LIBS})

# copy required images to every directory with executable
SET (IMAGES ${CMAKE_SOURCE_DIR}/images/objects.jpg)
FILE(COPY ${IMAGES} DESTINATION .)
FILE(COPY ${IMAGES} DESTINATION "Debug")
FILE(COPY ${IMAGES} DESTINATION "Release")
