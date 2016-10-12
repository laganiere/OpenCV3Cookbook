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
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

#include "visualTracker.h"

int main()
{
	// Create video procesor instance
	VideoProcessor processor;

	// generate the filename
	std::vector<std::string> imgs;
	std::string prefix = "goose/goose";
	std::string ext = ".bmp";

	for (long i = 130; i < 317; i++) {

		std::string name(prefix);
		std::ostringstream ss; ss << std::setfill('0') << std::setw(3) << i; name += ss.str();
		name += ext;

		std::cout << name << std::endl;
		imgs.push_back(name);
	}

	// Create feature tracker instance
	VisualTracker tracker;

	// Open video file
	processor.setInput(imgs);

	// set frame processor
	processor.setFrameProcessor(&tracker);

	// Declare a window to display the video
	processor.displayOutput("Tracked object");

	// Play the video at the original frame rate
	processor.setDelay(50);

	tracker.setBoundingBox(cv::Rect(290,100,65,40));
	// Start the process
	processor.run();

	cv::waitKey();
}