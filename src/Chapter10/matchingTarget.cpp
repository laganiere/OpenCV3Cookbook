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
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "targetMatcher.h"

int main()
{
	// Read input images
	cv::Mat target= cv::imread("cookbook1.bmp",0);
	cv::Mat image= cv::imread("book.jpg",0);
	if (!target.data || !image.data)
		return 0; 

    // Display the images
	cv::namedWindow("Target");
	cv::imshow("Target",target);

	// Prepare the matcher 
	TargetMatcher tmatcher("FAST","FREAK");
	tmatcher.setNormType(cv::NORM_HAMMING);

	// definition of the output data
	std::vector<cv::DMatch> matches;
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	std::vector<cv::Point2f> corners;
	// the reference image
	tmatcher.setTarget(target); 
	// match image with target
	tmatcher.detectTarget(image,corners,matches,keypoints1,keypoints2);
	// draw the target corners on the image
	cv::Point pt= cv::Point(corners[0]);
	cv::line(image,cv::Point(corners[0]),cv::Point(corners[1]),cv::Scalar(255,255,255),3);
	cv::line(image,cv::Point(corners[1]),cv::Point(corners[2]),cv::Scalar(255,255,255),3);
	cv::line(image,cv::Point(corners[2]),cv::Point(corners[3]),cv::Scalar(255,255,255),3);
	cv::line(image,cv::Point(corners[3]),cv::Point(corners[0]),cv::Scalar(255,255,255),3);

	cv::namedWindow("Image");
	cv::imshow("Image",image);

	// draw the matches
	cv::Mat imageMatches;
	cv::drawMatches(target,keypoints1,  // 1st image and its keypoints
		            image,keypoints2,  // 2nd image and its keypoints
					matches,			// the matches
					imageMatches,		// the image produced
					cv::Scalar(255,255,255),  // color of the lines
					cv::Scalar(255,255,255),  // color of the keypoints
					std::vector<char>(),
					2); 
	cv::namedWindow("Matches");
	cv::imshow("Matches",imageMatches);


/*	
	// Convert keypoints into Point2f	
	std::vector<cv::Point2f> points1, points2;
	
	for (std::vector<cv::DMatch>::const_iterator it= matches.begin();
			 it!= matches.end(); ++it) {

			 // Get the position of left keypoints
			 float x= keypoints1[it->queryIdx].pt.x;
			 float y= keypoints1[it->queryIdx].pt.y;
			 points1.push_back(cv::Point2f(x,y));
			 cv::circle(image1,cv::Point(x,y),3,cv::Scalar(255,255,255),3);
			 // Get the position of right keypoints
			 x= keypoints2[it->trainIdx].pt.x;
			 y= keypoints2[it->trainIdx].pt.y;
			 cv::circle(image2,cv::Point(x,y),3,cv::Scalar(255,255,255),3);
			 points2.push_back(cv::Point2f(x,y));
	}
	
	// Draw the epipolar lines
	std::vector<cv::Vec3f> lines1; 
	cv::computeCorrespondEpilines(cv::Mat(points1),1,fundemental,lines1);
		
	for (std::vector<cv::Vec3f>::const_iterator it= lines1.begin();
			 it!=lines1.end(); ++it) {

			 cv::line(image2,cv::Point(0,-(*it)[2]/(*it)[1]),
				             cv::Point(image2.cols,-((*it)[2]+(*it)[0]*image2.cols)/(*it)[1]),
							 cv::Scalar(255,255,255));
	}

	std::vector<cv::Vec3f> lines2; 
	cv::computeCorrespondEpilines(cv::Mat(points2),2,fundemental,lines2);
	
	for (std::vector<cv::Vec3f>::const_iterator it= lines2.begin();
		     it!=lines2.end(); ++it) {

			 cv::line(image1,cv::Point(0,-(*it)[2]/(*it)[1]),
				             cv::Point(image1.cols,-((*it)[2]+(*it)[0]*image1.cols)/(*it)[1]),
							 cv::Scalar(255,255,255));
	}

    // Display the images with epipolar lines
	cv::namedWindow("Right Image Epilines (RANSAC)");
	cv::imshow("Right Image Epilines (RANSAC)",image1);
	cv::namedWindow("Left Image Epilines (RANSAC)");
	cv::imshow("Left Image Epilines (RANSAC)",image2);
*/
	cv::waitKey();
	return 0;
}