/*------------------------------------------------------------------------------------------*\
   This file contains material supporting chapter 5 of the cookbook:
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
 
   Copyright (C) 2010-2011 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "mserFeatures.h"

int main()
{
	// Read input image
	cv::Mat image= cv::imread("building.jpg",0);
	if (!image.data)
		return 0; 
	// resize for book printing
	cv::resize(image, image, cv::Size(), 0.7, 0.7);

    // Display the image
	cv::namedWindow("Image");
	cv::imshow("Image",image);
	

	// basic MSER detector
	cv::MSER mser(5,     // delta value for local minima detection
		          200,   // min acceptable area 
				  1500); // max acceptable area

	// vector of point sets
    std::vector<std::vector<cv::Point> > points;
	// detect MSER features
	mser(image, points);

	std::cout << points.size() << " MSERs detected" << std::endl;

	// create white image
	cv::Mat output(image.size(),CV_8UC3);
	output= cv::Scalar(255,255,255);
	
	// random number generator
	cv::RNG rng;

	// for each detected feature
    for (std::vector<std::vector<cv::Point> >::iterator it= points.begin();
			   it!= points.end(); ++it) {

		// generate a random color
		cv::Vec3b c(rng.uniform(0,255),
			        rng.uniform(0,255),
					rng.uniform(0,255));

		std::cout << "MSER size= " << it->size() << std::endl;

		// for each point in MSER set
		for (std::vector<cv::Point>::iterator itPts= it->begin();
			        itPts!= it->end(); ++itPts) {

			//do not overwrite MSER pixels
			if (output.at<cv::Vec3b>(*itPts)[0]==255) {

				output.at<cv::Vec3b>(*itPts)= c;
			}
		}
	}

	cv::namedWindow("MSER point sets");
	cv::imshow("MSER point sets",output);
	cv::imwrite("mser.bmp", output);

	// detection using mserFeatures class

	// create MSER feature detector instance
	MSERFeatures mserF(200,  // min area 
		               1500, // max area
					   0.5); // ratio area threshold
	                         // default delta is used

	// the vector of bounding rotated rectangles
	std::vector<cv::RotatedRect> rects;

	// detect and get the image
	cv::Mat result= mserF.getImageOfEllipses(image,rects);

	// display detected MSER
	cv::namedWindow("MSER regions");
	cv::imshow("MSER regions",result);

	cv::waitKey();
}
