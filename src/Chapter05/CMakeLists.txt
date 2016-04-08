# cmake for OpenCV Cookbook 3rd edition chapter 05
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# add executable
add_executable( morphology morphology.cpp)
add_executable( morpho2 morpho2.cpp)
add_executable( mserFeatures mserFeatures.cpp)

# link libraries
target_link_libraries( morphology ${OpenCV_LIBS})
target_link_libraries( morpho2 ${OpenCV_LIBS})
target_link_libraries( mserFeatures ${OpenCV_LIBS})

# copy required images to every directory with executable
SET (IMAGES ${CMAKE_SOURCE_DIR}/images/binary.bmp 
			${CMAKE_SOURCE_DIR}/images/building.jpg
			${CMAKE_SOURCE_DIR}/images/group.jpg
			${CMAKE_SOURCE_DIR}/images/tower.jpg)
FILE(COPY ${IMAGES} DESTINATION .)
FILE(COPY ${IMAGES} DESTINATION "Debug")
FILE(COPY ${IMAGES} DESTINATION "Release")

