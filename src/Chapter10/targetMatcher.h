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

#if !defined TMATCHER
#define TMATCHER

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>

class TargetMatcher {

  private:

	  // pointer to the feature point detector object
	  cv::Ptr<cv::FeatureDetector> detector;
	  // pointer to the feature descriptor extractor object
	  cv::Ptr<cv::DescriptorExtractor> extractor;
	  cv::Mat target; // target image
	  int normType;
	  double distance; // min reprojection error

  public:

	  TargetMatcher(const std::string detectorName, const std::string descriptorName="") 
		  : normType(cv::NORM_L2), distance(1.0) {	  

		  if (detectorName.length()>0) {
			detector= cv::FeatureDetector::create(detectorName);

			if (descriptorName.length()>0) { 
				extractor= cv::DescriptorExtractor::create(descriptorName);
			} else { // or use the descriptor associated with the detector
				extractor= cv::DescriptorExtractor::create(detectorName);
			}
		  }
	  }

	  // Set the feature detector
	  void setFeatureDetector(const cv::Ptr<cv::FeatureDetector>& detect) {

		  detector= detect;
	  }

	  // Set descriptor extractor
	  void setDescriptorExtractor(const cv::Ptr<cv::DescriptorExtractor>& desc) {

		  extractor= desc;
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
	  }

	  // Identify good matches using RANSAC
	  // Return homography matrix and output matches
	  cv::Mat ransacTest(const std::vector<cv::DMatch>& matches,
		                 std::vector<cv::KeyPoint>& keypoints1, 
						 std::vector<cv::KeyPoint>& keypoints2,
					     std::vector<cv::DMatch>& outMatches) {

		// Convert keypoints into Point2f	
		std::vector<cv::Point2f> points1, points2;	
		
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
			CV_RANSAC,	  // RANSAC method
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

		  // find a RANSAC homography between target and image
	      cv::Mat homography= match(target,image,matches, 
		                            keypoints1, keypoints2);

		  // target corners
		  std::vector<cv::Point2f> corners;	
		  corners.push_back(cv::Point2f(0,0));
		  corners.push_back(cv::Point2f(target.cols-1,0));
		  corners.push_back(cv::Point2f(target.cols-1,target.rows-1));
		  corners.push_back(cv::Point2f(0,target.rows-1));

		  // reproject the target corners
		  cv::perspectiveTransform(corners,detectedCorners, homography);

		  return homography;
	  }
	  
	  // Match feature points using RANSAC
	  // returns homography matrix and output match set
	  cv::Mat match(const cv::Mat& image1, const cv::Mat& image2, // input images 
		  std::vector<cv::DMatch>& matches, // output matches and keypoints
		  std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2) { 
			  
		// 1. Detection of the feature points
		detector->detect(image1,keypoints1);
		detector->detect(image2,keypoints2);

		// 2. Extraction of the feature descriptors
		cv::Mat descriptors1, descriptors2;
		extractor->compute(image1,keypoints1,descriptors1);
		extractor->compute(image2,keypoints2,descriptors2);

		// 3. Match the two image descriptors
		//    (optionnaly apply some checking method)
   
		// Construction of the matcher with crosscheck 
		cv::BFMatcher matcher(normType,   // distance measure
	                          true);      // no crosscheck 
                             
		// match descriptors
	    std::vector<cv::DMatch> outputMatches;
		matcher.match(descriptors1,descriptors2,outputMatches);

		// 4. Validate matches using RANSAC
		cv::Mat homog= ransacTest(outputMatches, keypoints1, keypoints2, matches);

		// return the found fundemental matrix
		return homog;
	}
};

#endif
