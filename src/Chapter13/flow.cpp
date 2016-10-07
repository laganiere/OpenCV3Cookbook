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


#include <string>
#include <iostream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "videoprocessor.h"

// Drawing optical flow vectors on an image
void drawOptFlowMap(const cv::Mat& oflow, cv::Mat& flowImage, int stride, float scale, const cv::Scalar& color) {
	
	if (flowImage.size() != oflow.size()) {
		flowImage.create(oflow.size(), CV_8UC3);
		flowImage = cv::Vec3i(255,255,255);
	}

	for (int y = 0; y < oflow.rows; y += stride)
		for (int x = 0; x < oflow.cols; x += stride) {
				
			cv::Point2f vector = oflow.at< cv::Point2f>(y, x);
				
			cv::line(flowImage, cv::Point(x, y), 
				     cv::Point(static_cast<int>(x + scale*vector.x + 0.5), 
						       static_cast<int>(y + scale*vector.y + 0.5)), color);
				
			cv::circle(flowImage, cv::Point(static_cast<int>(x + scale*vector.x + 0.5),
				                            static_cast<int>(y + scale*vector.y + 0.5)), 1, color, -1);
		}
	
}

int main()
{
	// pick 2 frames of the sequence
	cv::Mat frame1= cv::imread("goose/goose230.bmp", 0);
	cv::Mat frame2= cv::imread("goose/goose237.bmp", 0);

	// Combined display
	cv::Mat combined(frame1.rows, frame1.cols + frame2.cols, CV_8U);
	frame1.copyTo(combined.colRange(0, frame1.cols));
	frame2.copyTo(combined.colRange(frame1.cols, frame1.cols+frame2.cols));
	cv::imshow("Frames", combined);

	// Create the optical flow algorithm
	cv::Ptr<cv::DualTVL1OpticalFlow> tvl1 = cv::createOptFlow_DualTVL1();

	std::cout << "regularization coeeficient: " << tvl1->getLambda() << std::endl; // the smaller the soomther
	std::cout << "Number of scales: " << tvl1->getScalesNumber() << std::endl; // number of scales
	std::cout << "Scale step: " << tvl1->getScaleStep() << std::endl; // size between scales
	std::cout << "Number of warpings: " << tvl1->getWarpingsNumber() << std::endl; // size between scales
	std::cout << "Stopping criteria: " << tvl1->getEpsilon() << " and " << tvl1->getOuterIterations() << std::endl; // size between scales
	

																													// compute the optical flow between 2 frames
	cv::Mat oflow; // image of 2D flow vectors
	tvl1->calc(frame1, frame2, oflow);

	// Draw the optical flow image
	cv::Mat flowImage;
	drawOptFlowMap(oflow,     // input flow vectors 
		flowImage, // image to be generated
		8,         // display vectors every 8 pixels
		2,         // multiply size of vectors by 2
		cv::Scalar(0, 0, 0)); // vector color

	cv::imshow("Optical Flow", flowImage);
	cv::waitKey();

	// compute the optical flow between 2 frames
	tvl1->calc(frame1, frame2, oflow);

	// Draw the optical flow image
	cv::Mat flowImage2;
	tvl1->setLambda(0.01);
	drawOptFlowMap(oflow,     // input flow vectors 
		flowImage2, // image to be generated
		8,         // display vectors every 8 pixels
		2,         // multiply size of vectors by 2
		cv::Scalar(0, 0, 0)); // vector color

	cv::imshow("Optical Flow (2)", flowImage2);
	cv::waitKey();

	//	calcOpticalFlowSF(frame1, frame2,
	//	flow,
		// 3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);
	// flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

	/*
	// Create video procesor instance
	VideoProcessor processor;

	// generate the filename
	std::vector<std::string> imgs;
	std::string prefix = "goose/goose";
	std::string ext = ".bmp";

	for (long i = 0; i < 317; i++) {

		std::string name(prefix);
		std::ostringstream ss; ss << std::setfill('0') << std::setw(3) << i; name += ss.str(); 
		name += ext;

		std::cout << name << std::endl;
		imgs.push_back(name);
	}
	cv::waitKey();
		// Open video file
//	processor.setInput(imgs);
	processor.setInput("goose.mp4");

	// set frame processor
//	processor.setFrameProcessor(&segmentor);

	// Declare a window to display the video
	processor.displayOutput("Original video");

	// Play the video at the original frame rate
	processor.setDelay(50);

	// Start the process
	processor.run();

	cv::waitKey(); */
}