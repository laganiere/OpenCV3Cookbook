/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 3 of the book:
OpenCV3 Computer Vision Application Programming Cookbook
Third Edition
by Robert Laganiere, Packt Publishing, 2016.

This program is free software; permission is hereby granted to use, copy, modify,
and distribute this source code, or portions thereof, for any purpose, without fee,
subject to the restriction that the copyright notice may not be removed
or altered from any source or altered source distribution.
The software is released on an as-is basis and without any warranties of any kind.
In particular, the software is not guaranteed to be fault-tolerant or free from failure.
The author disclaims all warranties with regard to this software, any use,
and any consequent failure, is purely the responsibility of the user.

Copyright (C) 2016 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "colordetector.h"

int main()
{
    // 1. Create image processor object
	ColorDetector cdetect;

    // 2. Read input image
	cv::Mat image= cv::imread("boldt.jpg");
	if (image.empty())
		return 0; 
	cv::namedWindow("Original Image");
	cv::imshow("Original Image", image);

    // 3. Set input parameters
	cdetect.setTargetColor(230,190,130); // here blue sky

    // 4. Process the image and display the result
	cv::namedWindow("result");
	cv::Mat result = cdetect.process(image);
	cv::imshow("result",result);

	// or using functor
	// here distance is measured with the Lab color space
	ColorDetector colordetector(230, 190, 130,  // color
		                             45, true); // Lab threshold
	cv::namedWindow("result (functor)");
	result = colordetector(image);
	cv::imshow("result (functor)",result);

	// testing floodfill
	cv::floodFill(image,            // input/ouput image
		cv::Point(100, 50),         // seed point
		cv::Scalar(255, 255, 255),  // repainted color
		(cv::Rect*)0,  // bounding rectangle of the repainted pixel set
		cv::Scalar(35, 35, 35),     // low and high difference threshold
		cv::Scalar(35, 35, 35),     // most of the time will be identical
		cv::FLOODFILL_FIXED_RANGE); // pixels are compared to seed color

	cv::namedWindow("Flood Fill result");
	result = colordetector(image);
	cv::imshow("Flood Fill result", image);

	cv::waitKey();

	return 0;
}

