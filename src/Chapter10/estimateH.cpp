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
#include <opencv2/nonfree/features2d.hpp>

int main()
{
	// Read input images
	cv::Mat image1= cv::imread("parliament1.bmp",0);
	cv::Mat image2= cv::imread("parliament2.bmp",0);
	if (!image1.data || !image2.data)
		return 0; 

    // Display the images
	cv::namedWindow("Image 1");
	cv::imshow("Image 1",image1);
	cv::namedWindow("Image 2");
	cv::imshow("Image 2",image2);
	
	// 0. Construction of the detector and descriptor
	// for example SURF
	cv::Ptr<cv::FeatureDetector> detector= new cv::SURF();
	cv::Ptr<cv::DescriptorExtractor> extractor= detector;

	// 1. Detection of the feature points
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;
	detector->detect(image1,keypoints1);
	detector->detect(image2,keypoints2);

	std::cout << "Number of feature points (1): " << keypoints1.size() << std::endl;
	std::cout << "Number of feature points (2): " << keypoints2.size() << std::endl;

	// 2. Extraction of the feature descriptors
	cv::Mat descriptors1, descriptors2;
	extractor->compute(image1,keypoints1,descriptors1);
	extractor->compute(image2,keypoints2,descriptors2);

	// 3. Match the two image descriptors
   
	// Construction of the matcher with crosscheck 
	cv::BFMatcher matcher(cv::NORM_L2, true);                            
	// matching
	std::vector<cv::DMatch> matches;
	matcher.match(descriptors1,descriptors2,matches);

	// draw the matches
	cv::Mat imageMatches;
	cv::drawMatches(image1,keypoints1,  // 1st image and its keypoints
		            image2,keypoints2,  // 2nd image and its keypoints
					matches,			// the matches
					imageMatches,		// the image produced
					cv::Scalar(255,255,255),  // color of the lines
					cv::Scalar(255,255,255),  // color of the keypoints
					std::vector<char>(),
					2); 
	cv::namedWindow("Matches");
	cv::imshow("Matches",imageMatches);
	
	// Convert keypoints into Point2f
	std::vector<cv::Point2f> points1, points2;
	for (std::vector<cv::DMatch>::const_iterator it= matches.begin();
		 it!= matches.end(); ++it) {

			 // Get the position of left keypoints
			 float x= keypoints1[it->queryIdx].pt.x;
			 float y= keypoints1[it->queryIdx].pt.y;
			 points1.push_back(cv::Point2f(x,y));
			 // Get the position of right keypoints
			 x= keypoints2[it->trainIdx].pt.x;
			 y= keypoints2[it->trainIdx].pt.y;
			 points2.push_back(cv::Point2f(x,y));
	}

	std::cout << points1.size() << " " << points2.size() << std::endl; 

	// Find the homography between image 1 and image 2
	std::vector<uchar> inliers(points1.size(),0);
	cv::Mat homography= cv::findHomography(
		points1,points2, // corresponding points
		inliers,	// outputed inliers matches 
		CV_RANSAC,	// RANSAC method
		1.);	    // max distance to reprojection point

	// Draw the inlier points
	std::vector<cv::Point2f>::const_iterator itPts= points1.begin();
	std::vector<uchar>::const_iterator itIn= inliers.begin();
	while (itPts!=points1.end()) {

		// draw a circle at each inlier location
		if (*itIn) 
 			cv::circle(image1,*itPts,3,cv::Scalar(255,255,255));
		
		++itPts;
		++itIn;
	}

	itPts= points2.begin();
	itIn= inliers.begin();
	while (itPts!=points2.end()) {

		// draw a circle at each inlier location
		if (*itIn) 
			cv::circle(image2,*itPts,3,cv::Scalar(255,255,255));
		
		++itPts;
		++itIn;
	}

    // Display the images with points
	cv::namedWindow("Image 1 Homography Points");
	cv::imshow("Image 1 Homography Points",image1);
	cv::namedWindow("Image 2 Homography Points");
	cv::imshow("Image 2 Homography Points",image2);

	// Warp image 1 to image 2
	cv::Mat result;
	cv::warpPerspective(image1, // input image
		result,			// output image
		homography,		// homography
		cv::Size(2*image1.cols,image1.rows)); // size of output image

	// Copy image 1 on the first half of full image
	cv::Mat half(result,cv::Rect(0,0,image2.cols,image2.rows));
	image2.copyTo(half);

    // Display the warp image
	cv::namedWindow("Image mosaic");
	cv::imshow("Image mosaic",result);

	cv::waitKey();
	return 0;
}