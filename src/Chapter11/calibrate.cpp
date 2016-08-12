/*------------------------------------------------------------------------------------------*\
   This file contains material supporting chapter 10 of the cookbook:  
   Computer Vision Programming using the OpenCV Library 
   Second Edition 
   by Robert Laganiere, Packt Publishing, 2013.

   This program is free software; permission is hereby granted to use, copy, modify, 
   and distribute this source code, or portions thereof, for any purpose, without fee, 
   subject to the restriction that the copyright notice may not be removed 
   or altered from any source or altered source distribution. 
   The software is released on an as-is basis and without any warranties of any kind. 
   In particular, the software is not guaranteed to be fault-tolerant or free from failure. 
   The author disclaims all warranties with regard to this software, any use, 
   and any consequent failure, is purely the responsibility of the user.
 
   Copyright (C) 2013 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/

#include <iostream>
#include <iomanip>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "CameraCalibrator.h"

	// Applies a lookup table transforming an input image into a 1-channel image
	cv::Mat applyLookUp(const cv::Mat& image, const cv::MatND& lookup) {

		// Set output image (always 1-channel)
		cv::Mat result(image.rows,image.cols,CV_8U);
		cv::Mat_<uchar>::iterator itr= result.begin<uchar>();

		// Iterates over the input image
		cv::Mat_<uchar>::const_iterator it= image.begin<uchar>();
		cv::Mat_<uchar>::const_iterator itend= image.end<uchar>();

		// Applies lookup to each pixel
		for ( ; it!= itend; ++it, ++itr) {

			*itr= lookup.at<uchar>(*it);
		}
		return result;
	}

int main()
{
	cv::namedWindow("Board Image");
	cv::Mat image;
	std::vector<std::string> filelist;

	// generate list of chessboard image filename
	for (int i=1; i<=20; i++) {

		std::stringstream str;
		str << "chessboards/chessboard" << std::setw(2) << std::setfill('0') << i << ".jpg";
		std::cout << str.str() << std::endl;

		filelist.push_back(str.str());
		image= cv::imread(str.str(),0);
		cv::imshow("Board Image",image);
	
		 cv::waitKey(100);
	}

	// Create calibrator object
    CameraCalibrator cameraCalibrator;
	// add the corners from the chessboard
	cv::Size boardSize(6,4);
	cameraCalibrator.addChessboardPoints(
		filelist,	// filenames of chessboard image
		boardSize, "Detected points");	// size of chessboard

	// calibrate the camera
    //	cameraCalibrator.setCalibrationFlag(true,true);
	cameraCalibrator.calibrate(image.size());

    // Image Undistortion
    image = cv::imread(filelist[6],0);
	cv::Mat uImage= cameraCalibrator.remap(image);

	// display camera matrix
	cv::Mat cameraMatrix= cameraCalibrator.getCameraMatrix();
	std::cout << " Camera intrinsic: " << cameraMatrix.rows << "x" << cameraMatrix.cols << std::endl;
	std::cout << cameraMatrix.at<double>(0,0) << " " << cameraMatrix.at<double>(0,1) << " " << cameraMatrix.at<double>(0,2) << std::endl;
	std::cout << cameraMatrix.at<double>(1,0) << " " << cameraMatrix.at<double>(1,1) << " " << cameraMatrix.at<double>(1,2) << std::endl;
	std::cout << cameraMatrix.at<double>(2,0) << " " << cameraMatrix.at<double>(2,1) << " " << cameraMatrix.at<double>(2,2) << std::endl;

	cv::namedWindow("Original Image");
    cv::imshow("Original Image", image);
	cv::namedWindow("Undistorted Image");
    cv::imshow("Undistorted Image", uImage);

	cv::waitKey();
	return 0;

	// Create an image inversion table
	int dim(256);
	cv::Mat lut(1,  // 1 dimension
		&dim,       // 256 entries
		CV_8U);     // uchar

	for (int i=0; i<256; i++) {
		
		lut.at<uchar>(i)= 255-i;
	}

	// Apply lookup and display negative image
	cv::namedWindow("Negative image");
//	cv::imshow("Negative image",h.applyLookUp(image,lookup));
	cv::imshow("Negative image",applyLookUp(image,lut));


	cv::waitKey();
	return 0;
}
