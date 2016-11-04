# cmake for OpenCV Cookbook 3rd edition chapter 14
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# add executable
add_executable( recognizeFace recognizeFace.cpp)
add_executable( trainCascade trainCascade.cpp)
add_executable( detectFaces detectFaces.cpp)
add_executable( trainSVM trainSVM.cpp)
add_executable( detectPeople detectPeople.cpp)
add_executable( bagWords bagWords.cpp)

# link libraries
target_link_libraries( recognizeFace ${OpenCV_LIBS})
target_link_libraries( trainCascade ${OpenCV_LIBS})
target_link_libraries( detectFaces ${OpenCV_LIBS})
target_link_libraries( trainSVM ${OpenCV_LIBS})
target_link_libraries( detectPeople ${OpenCV_LIBS})
target_link_libraries( bagWords ${OpenCV_LIBS})

# copy required images to every directory with executable
SET (IMAGES ${CMAKE_SOURCE_DIR}/images/girl.jpg
            ${CMAKE_SOURCE_DIR}/images/face0_1.png
            ${CMAKE_SOURCE_DIR}/images/face0_2.png
            ${CMAKE_SOURCE_DIR}/images/face1_1.png
            ${CMAKE_SOURCE_DIR}/images/face1_2.png
)
FILE(COPY ${IMAGES} DESTINATION .)
FILE(COPY ${IMAGES} DESTINATION "Debug")
FILE(COPY ${IMAGES} DESTINATION "Release")
FILE(COPY ${CMAKE_SOURCE_DIR}/images/stopSamples/ DESTINATION ./stopSamples/)
FILE(COPY ${CMAKE_SOURCE_DIR}/images/stopSamples/ DESTINATION "Debug/stopSamples/")
FILE(COPY ${CMAKE_SOURCE_DIR}/images/stopSamples/ DESTINATION "Release/stopSamples/")


