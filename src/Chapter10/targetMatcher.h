/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 10 of the book:
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

#if !defined TMATCHER
#define TMATCHER

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

class TargetMatcher {

  private:

	  // pointer to the feature point detector object
	  cv::Ptr<cv::FeatureDetector> detector;
	  // pointer to the feature descriptor extractor object
	  cv::Ptr<cv::DescriptorExtractor> descriptor;
	  cv::Mat target; // target image
	  int normType;
	  double distance; // min reprojection error
	  int numberOfLevels;
	  double scaleFactor;
	  std::vector<cv::Mat> pyramid;

	  // create a pyramid of target images
	  void createPyramid() {

		  pyramid.clear();
		  cv::Mat layer(target);
		  for (int i = 0; i < numberOfLevels; i++) {
			  pyramid.push_back(target.clone());
			  resize(target, target, cv::Size(), scaleFactor, scaleFactor);
		  }
	  }

  public:

	  TargetMatcher(const cv::Ptr<cv::FeatureDetector> &detector,
			  const cv::Ptr<cv::DescriptorExtractor> &descriptor = cv::Ptr<cv::DescriptorExtractor>(),
		      int numberOfLevels=8, double scaleFactor=0.9)
		  : detector(detector), descriptor(descriptor), normType(cv::NORM_L2), distance(1.0),
		    numberOfLevels(numberOfLevels), scaleFactor(scaleFactor) {

		  // in this case use the associated descriptor
		  if (!this->descriptor) {
			  this->descriptor = this->detector;
		  }

	  }


	  // Set the feature detector
	  void setFeatureDetector(const cv::Ptr<cv::FeatureDetector>& detect) {

		  detector= detect;
	  }

	  // Set descriptor extractor
	  void setDescriptorExtractor(const cv::Ptr<cv::DescriptorExtractor>& desc) {

		  descriptor = desc;
	  }

	  // Set the norm to be used for matching
	  void setNormType(int norm) {

		  normType= norm;
	  }

	  // Set the minimum reprojection distance
	  void setReprojectionDistance(double d) {

		  distance= d;
	  }

	  // Set the target image
	  void setTarget(const cv::Mat t) {

		  target= t;
		  createPyramid();
	  }

	  // Identify good matches using RANSAC
	  // Return homography matrix and output matches
	  cv::Mat ransacTest(const std::vector<cv::DMatch>& matches,
		                 std::vector<cv::KeyPoint>& keypoints1, 
						 std::vector<cv::KeyPoint>& keypoints2,
					     std::vector<cv::DMatch>& outMatches) {

		// Convert keypoints into Point2f	
		std::vector<cv::Point2f> points1, points2;	
		outMatches.clear();
		
		for (std::vector<cv::DMatch>::const_iterator it= matches.begin();
			 it!= matches.end(); ++it) {

			 // Get the position of left keypoints
			 points1.push_back(keypoints1[it->queryIdx].pt);
			 // Get the position of right keypoints
			 points2.push_back(keypoints2[it->trainIdx].pt);
	    }

		// Find the homography between image 1 and image 2
		std::vector<uchar> inliers(points1.size(),0);
		cv::Mat homography= cv::findHomography(
			points1,points2, // corresponding points
		    inliers,      // match status (inlier or outlier)  
			cv::RHO,	  // RHO method
		    distance);    // max distance to reprojection point
	
		// extract the surviving (inliers) matches
		std::vector<uchar>::const_iterator itIn= inliers.begin();
		std::vector<cv::DMatch>::const_iterator itM= matches.begin();
		// for all matches
		for ( ;itIn!= inliers.end(); ++itIn, ++itM) {

			if (*itIn) { // it is a valid match

				outMatches.push_back(*itM);
			}
		}

		return homography;
	  }

	  // detect the defined planar target in an image
	  // returns the homography 
	  // the 4 corners of the detected target
	  // plus matches and keypoints
	  cv::Mat detectTarget(const cv::Mat& image, 
		  // position of the target corners (clock-wise)
		  std::vector<cv::Point2f>& detectedCorners, 		  
		  std::vector<cv::DMatch>& matches,
		  std::vector<cv::KeyPoint>& keypoints1,
		  std::vector<cv::KeyPoint>& keypoints2) {

		  cv::Mat bestHomography;
		  cv::Size bestSize;
		  int maxInliers = 0;
		  cv::Mat homography;
		  for (auto it = pyramid.begin(); it != pyramid.end(); ++it) {
			  // find a RANSAC homography between target and image
			  homography = match(*it, image, matches,
										 keypoints1, keypoints2);

			  if (matches.size() > maxInliers) { // we have a better H
				  maxInliers = matches.size();
				  bestHomography = homography;
				  bestSize = it->size();
			  }
		  }

		  if (maxInliers > 8) { // the estimate is valid

			// target corners
			  std::vector<cv::Point2f> corners;
			  corners.push_back(cv::Point2f(0, 0));
			  corners.push_back(cv::Point2f(bestSize.width - 1, 0));
			  corners.push_back(cv::Point2f(bestSize.width - 1, bestSize.height - 1));
			  corners.push_back(cv::Point2f(0, bestSize.height - 1));

			  // reproject the target corners
			  cv::perspectiveTransform(corners, detectedCorners, bestHomography);
		  }

		  std::cout << "Best number of inliers= " << maxInliers << std::endl;
 
		  return bestHomography;
	  }
	  
	  // Match feature points using RANSAC
	  // returns homography matrix and output match set
	  cv::Mat match(const cv::Mat& image1, const cv::Mat& image2, // input images 
		  std::vector<cv::DMatch>& matches, // output matches and keypoints
		  std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2) { 
			  
		// 1. Detection of the feature points
		keypoints1.clear();
		keypoints2.clear();
		detector->detect(image1,keypoints1);
		detector->detect(image2,keypoints2);

		std::cout << "Interest points: target=" <<keypoints1.size() << " image=" << keypoints2.size() << std::endl;

		// 2. Extraction of the feature descriptors
		cv::Mat descriptors1, descriptors2;
		descriptor->compute(image1,keypoints1,descriptors1);
		descriptor->compute(image2,keypoints2,descriptors2);

		// 3. Match the two image descriptors
		//    (optionnaly apply some checking method)
   
		// Construction of the matcher with crosscheck 
		cv::BFMatcher matcher(normType);   // distance measure
                             
		// match descriptors
	    std::vector<cv::DMatch> outputMatches;
		matcher.match(descriptors1,descriptors2,outputMatches);
        std::cout << "Number of matches=" << outputMatches.size() << std::endl;
		// 4. Validate matches using RANSAC
		cv::Mat homog= ransacTest(outputMatches, keypoints1, keypoints2, matches);
		std::cout << "Number of inliers=" << matches.size() << std::endl;

		// return the found fundemental matrix
		return homog;
	}
};

#endif
