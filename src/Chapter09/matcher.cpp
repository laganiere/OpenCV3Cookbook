/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 9 of the book:
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
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/features2d.hpp>

int main()
{
	// image matching

	// 1. Read input images
	cv::Mat image1= cv::imread("church01.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat image2= cv::imread("church02.jpg",CV_LOAD_IMAGE_GRAYSCALE);

	// 2. Define keypoints vector
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;

	// 3. Define feature detector
	// Construct the SURF feature detector object
	cv::Ptr<cv::FeatureDetector> detector = new cv::SURF(1500);

	// 4. Keypoint detection
	// Detect the SURF features
	detector->detect(image1,keypoints1);
	detector->detect(image2,keypoints2);

	// Draw feature points
	cv::Mat featureImage;
	cv::drawKeypoints(image1,keypoints1,featureImage,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Display the corners
	cv::namedWindow("SURF");
	cv::imshow("SURF",featureImage);

	std::cout << "Number of SURF keypoints (image 1): " << keypoints1.size() << std::endl; 
	std::cout << "Number of SURF keypoints (image 2): " << keypoints2.size() << std::endl; 

	// SURF includes both the detector and descriptor extractor
	cv::Ptr<cv::DescriptorExtractor> descriptor = detector;

	// 5. Extract the descriptor
    cv::Mat descriptors1;
    cv::Mat descriptors2;
    descriptor->compute(image1,keypoints1,descriptors1);
    descriptor->compute(image2,keypoints2,descriptors2);

   // Construction of the matcher 
   cv::BFMatcher matcher(cv::NORM_L2);
   // Match the two image descriptors
   std::vector<cv::DMatch> matches;
   matcher.match(descriptors1,descriptors2, matches);

   // draw matches
   cv::Mat imageMatches;
   cv::drawMatches(
     image1,keypoints1, // 1st image and its keypoints
     image2,keypoints2, // 2nd image and its keypoints
     matches,            // the matches
     imageMatches,      // the image produced
     cv::Scalar(255,255,255),  // color of lines
     cv::Scalar(255,255,255)); // color of points

    // Display the image of matches
	cv::namedWindow("SURF Matches");
	cv::imshow("SURF Matches",imageMatches);

	std::cout << "Number of matches: " << matches.size() << std::endl; 

	// radius match
	float maxDist= 0.4;
    std::vector<std::vector<cv::DMatch> > matches2;
	matcher.radiusMatch(descriptors1, descriptors2, matches2, 
		                maxDist); // maximum acceptable distance
	                              // between the 2 descriptors
   cv::drawMatches(
     image1,keypoints1, // 1st image and its keypoints
     image2,keypoints2, // 2nd image and its keypoints
     matches2,          // the matches
     imageMatches,      // the image produced
     cv::Scalar(255,255,255),  // color of lines
     cv::Scalar(255,255,255)); // color of points

    int nmatches=0;
	for (int i=0; i< matches2.size(); i++) nmatches+= matches2[i].size();
	std::cout << "Number of matches (with min radius): " << nmatches << std::endl; 

    // Display the image of matches
	cv::namedWindow("SURF Matches (with min radius)");
	cv::imshow("SURF Matches (with min radius)",imageMatches);

	// perform the ratio test

	// find the best two matches of each keypoint
    matcher.knnMatch(descriptors1,descriptors2, matches2, 
		             2); // find the k best matches
	matches.clear();

	// perform ratio test
	double ratio= 0.85;
    std::vector<std::vector<cv::DMatch> >::iterator it;
	for (it= matches2.begin(); it!= matches2.end(); ++it) {

		//   first best match/second best match
		if ((*it)[0].distance/(*it)[1].distance < ratio) {
			// it is an acceptable match
			matches.push_back((*it)[0]);
		}
	}
	// matches is the new match set

   cv::drawMatches(
     image1,keypoints1, // 1st image and its keypoints
     image2,keypoints2, // 2nd image and its keypoints
     matches,            // the matches
     imageMatches,      // the image produced
     cv::Scalar(255,255,255),  // color of lines
     cv::Scalar(255,255,255)); // color of points

	std::cout << "Number of matches (after ratio test): " << matches.size() << std::endl; 

    // Display the image of matches
	cv::namedWindow("SURF Matches (ratio test)");
	cv::imshow("SURF Matches (ratio test)",imageMatches);

   // Construction of the matcher with crosscheck 
   cv::BFMatcher matcher2(cv::NORM_L2, //distance measure
	                     true);        // crosscheck flag
   // Match the two image descriptors
   matcher2.match(descriptors1,descriptors2, matches);

   // draw matches
   cv::drawMatches(
     image1,keypoints1, // 1st image and its keypoints
     image2,keypoints2, // 2nd image and its keypoints
     matches,            // the matches
     imageMatches,      // the image produced
     cv::Scalar(255,255,255),  // color of lines
     cv::Scalar(255,255,255)); // color of points

   // Display the image of matches
   cv::namedWindow("SURF Matches (with crosscheck)");
   cv::imshow("SURF Matches (with crosscheck)",imageMatches);

   std::cout << "Number of matches (crosscheck): " << matches.size() << std::endl; 

   // SIFT
   // 3. Define feature detector
	 
   // Construct the SURF feature detector object
   detector = new cv::SIFT(500);
   
	// 4. Keypoint detection
	// Detect the SIFT features
	detector->detect(image1,keypoints1);
	detector->detect(image2,keypoints2);

	// Draw feature points
	cv::drawKeypoints(image1,keypoints1,featureImage,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Display the corners
	cv::namedWindow("SIFT");
	cv::imshow("SIFT",featureImage);

	std::cout << "Number of SIFT keypoints (image 1): " << keypoints1.size() << std::endl; 
	std::cout << "Number of SIFT keypoints (image 2): " << keypoints2.size() << std::endl; 

    // Display the image of matches
	cv::namedWindow("Keypoints (image 1)");
	cv::imshow("Keypoints (image 1)",image1);

	// SIFT includes both the detector and descriptor extractor
	descriptor = detector;

	// 5. Extract the descriptor
    descriptor->compute(image1,keypoints1,descriptors1);
    descriptor->compute(image2,keypoints2,descriptors2);

   // Match the two image descriptors
   matcher.match(descriptors1,descriptors2, matches);

   // draw matches
   cv::drawMatches(
     image1,keypoints1, // 1st image and its keypoints
     image2,keypoints2, // 2nd image and its keypoints
     matches,            // the matches
     imageMatches,      // the image produced
     cv::Scalar(255,255,255),  // color of lines
     cv::Scalar(255,255,255)); // color of points

    // Display the image of matches
	cv::namedWindow("SIFT Matches");
	cv::imshow("SIFT Matches",imageMatches);

	std::cout << "Number of matches: " << matches.size() << std::endl; 

	// scale-invariance test

	// Read input images
	image1= cv::imread("church01.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	image2= cv::imread("church03.jpg",CV_LOAD_IMAGE_GRAYSCALE);

	// Keypoint detection
	// Detect the SIFT features
	detector->detect(image1,keypoints1);
	detector->detect(image2,keypoints2);

	// Draw feature points
	cv::drawKeypoints(image1,keypoints1,featureImage,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Display the corners
	cv::namedWindow("SIFT");
	cv::imshow("SIFT",featureImage);

	std::cout << "Number of SIFT keypoints (image 1): " << keypoints1.size() << std::endl; 
	std::cout << "Number of SIFT keypoints (image 2): " << keypoints2.size() << std::endl; 

	// Extract the descriptor
    descriptor->compute(image1,keypoints1,descriptors1);
    descriptor->compute(image2,keypoints2,descriptors2);

    // Match the two image descriptors
    matcher.match(descriptors1,descriptors2, matches);

	// extract the 50 best matches
	std::nth_element(matches.begin(),matches.begin()+50,matches.end());
	matches.erase(matches.begin()+50,matches.end());

   // draw matches
	cv::drawMatches(
		image1, keypoints1, // 1st image and its keypoints
		image2, keypoints2, // 2nd image and its keypoints
		matches,            // the matches
		imageMatches,      // the image produced
		cv::Scalar(255, 255, 255),  // color of lines
		cv::Scalar(255, 255, 255), // color of points
		std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // Display the image of matches
	cv::namedWindow("multi-scale SIFT Matches");
	cv::imshow("multi-scale SIFT Matches",imageMatches);

	std::cout << "Number of matches: " << matches.size() << std::endl; 

   cv::waitKey();
   return 0;
}
