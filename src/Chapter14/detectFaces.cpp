/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 13 of the book:
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

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

int main()
{
	cv::Mat inputImage = cv::imread("stopSamples/stop9.jpg", cv::IMREAD_GRAYSCALE);
		

	cv::CascadeClassifier cascade;
	if (!cascade.load("stopSamples/classifier/cascade.xml")) { printf("--(!)Error loading face cascade\n"); return -1; };
	// predict the label of this image

	std::vector<cv::Rect> detections;

	cascade.detectMultiScale(inputImage, detections, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(24, 24), cv::Size(128, 128));

	std::cout << "detections= " << detections.size() << std::endl;
	for (int i = 0; i < detections.size(); i++)
		cv::rectangle(inputImage, detections[i], cv::Scalar(255, 255, 255), 2);

	cv::imshow("Input image", inputImage);
	cv::waitKey();
}