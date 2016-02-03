/*------------------------------------------------------------------------------------------*\
   This file contains material supporting chapter 2 of the cookbook:  
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


#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main()
{
	cv::Mat image1;
	cv::Mat image2;

	image1= cv::imread("boldt.jpg");
	image2= cv::imread("rain.jpg");
	if (!image1.data)
		return 0; 
	if (!image2.data)
		return 0; 

	// images ares resize for book printing
	cv::resize(image1, image1, cv::Size(), 0.6, 0.6);
	cv::resize(image2, image2, cv::Size(), 0.6, 0.6);

	cv::namedWindow("Image 1");
	cv::imshow("Image 1",image1);
	cv::namedWindow("Image 2");
	cv::imshow("Image 2",image2);

	cv::Mat result;
	// add two images
	cv::addWeighted(image1,0.7,image2,0.9,0.,result);

	cv::namedWindow("result");
	cv::imshow("result",result);

	// using overloaded operator
	result= 0.7*image1+0.9*image2;

	cv::namedWindow("result with operators");
	cv::imshow("result with operators",result);

	image2= cv::imread("rain.jpg",0);
	// images ares resize for book printing
	cv::resize(image2, image2, cv::Size(), 0.6, 0.6);

	// create vector of 3 images
	std::vector<cv::Mat> planes;
	// split 1 3-channel image into 3 1-channel images
	cv::split(image1,planes);
	// add to blue channel
	planes[0]+= image2;
	// merge the 3 1-channel images into 1 3-channel image
	cv::merge(planes,result);

	cv::namedWindow("Result on blue channel");
	cv::imshow("Result on blue channel",result);

	cv::waitKey();

	return 0;
}
