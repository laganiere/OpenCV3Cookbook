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
 
   Copyright (C) 2013 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/

#if !defined MORPHOF
#define MORPHOF

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class MorphoFeatures {

  private:

	  // threshold to produce binary image
	  int threshold;
	  // structuring elements used in corner detection
	  cv::Mat_<uchar> cross;
	  cv::Mat_<uchar> diamond;
	  cv::Mat_<uchar> square;
	  cv::Mat_<uchar> x;

	  void applyThreshold(cv::Mat& result)  {

          // Apply threshold on result
		  if (threshold>0)
			cv::threshold(result, result, threshold, 255, cv::THRESH_BINARY_INV);
	  }

  public:

	  MorphoFeatures() : threshold(-1), 
		  cross(5, 5), diamond(5, 5), square(5, 5), x(5, 5) {
	
		  // Creating the cross-shaped structuring element
		  cross <<
			  0, 0, 1, 0, 0,
			  0, 0, 1, 0, 0,
			  1, 1, 1, 1, 1,
			  0, 0, 1, 0, 0,
			  0, 0, 1, 0, 0;
		  
		  // Creating the diamond-shaped structuring element
		  diamond <<
			  0, 0, 1, 0, 0,
			  0, 1, 1, 1, 0,
			  1, 1, 1, 1, 1,
			  0, 1, 1, 1, 0,
			  0, 0, 1, 0, 0;
		  
		  // Creating the x-shaped structuring element
		  x <<
			  1, 0, 0, 0, 1,
			  0, 1, 0, 1, 0,
			  0, 0, 1, 0, 0,
			  0, 1, 0, 1, 0,
			  1, 0, 0, 0, 1;

		  // Creating the square-shaped structuring element
		  x <<
			  1, 1, 1, 1, 1,
			  1, 1, 1, 1, 1,
			  1, 1, 1, 1, 1,
			  1, 1, 1, 1, 1,
			  1, 1, 1, 1, 1;
	  }

	  void setThreshold(int t) {

		  threshold= t;
	  }

	  int getThreshold() const {

		  return threshold;
	  }

	  cv::Mat getEdges(const cv::Mat &image) {

		  // Get the gradient image
		  cv::Mat result;
		  cv::morphologyEx(image,result,cv::MORPH_GRADIENT,cv::Mat());

          // Apply threshold to obtain a binary image
		  applyThreshold(result);

		  return result;
	  }

	  cv::Mat getCorners(const cv::Mat &image) {

		  cv::Mat result;

		  // Dilate with a cross	
		  cv::dilate(image,result,cross);

		  // Erode with a diamond
		  cv::erode(result,result,diamond);

		  cv::Mat result2;
		  // Dilate with a X	
		  cv::dilate(image,result2,x);

		  // Erode with a square
		  cv::erode(result2,result2,square);

		  // Corners are obtained by differencing
		  // the two closed images
		  cv::absdiff(result2,result,result);

          // Apply threshold to obtain a binary image
		  applyThreshold(result);

		  return result;
	  }

	  void drawOnImage(const cv::Mat& binary, cv::Mat& image) {
		  	  
		  cv::Mat_<uchar>::const_iterator it= binary.begin<uchar>();
		  cv::Mat_<uchar>::const_iterator itend= binary.end<uchar>();

		  // for each pixel	
		  for (int i=0; it!= itend; ++it,++i) {
			  if (!*it)
				  cv::circle(image,cv::Point(i%image.step,i/image.step),5,cv::Scalar(255,0,0));
		  }
	  }
};


#endif
