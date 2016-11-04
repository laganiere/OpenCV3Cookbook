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
	// open the positive sample images
	std::vector<cv::Mat> referenceImages;
	referenceImages.push_back(cv::imread("stopSamples/stop00.png"));
	referenceImages.push_back(cv::imread("stopSamples/stop01.png"));
	referenceImages.push_back(cv::imread("stopSamples/stop02.png"));
	referenceImages.push_back(cv::imread("stopSamples/stop03.png"));
	referenceImages.push_back(cv::imread("stopSamples/stop04.png"));
	referenceImages.push_back(cv::imread("stopSamples/stop05.png"));
	referenceImages.push_back(cv::imread("stopSamples/stop06.png"));
	referenceImages.push_back(cv::imread("stopSamples/stop07.png"));

	// create a composite image
	cv::Mat positveImages(2 * referenceImages[0].rows, 4 * referenceImages[0].cols, CV_8UC3);
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 4; j++) {

			referenceImages[i * 2 + j].copyTo(positveImages(cv::Rect(j*referenceImages[i * 2 + j].cols, i*referenceImages[i * 2 + j].rows, referenceImages[i * 2 + j].cols, referenceImages[i * 2 + j].rows)));
		}

	cv::imshow("Positive samples", positveImages);

	cv::Mat negative = cv::imread("stopSamples/bg01.jpg");
	cv::resize(negative, negative, cv::Size(), 0.33, 0.33);
	cv::imshow("One negative sample", negative);

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