/*------------------------------------------------------------------------------------------*\
   This file contains material supporting chapter 3 of the cookbook:  
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

#if !defined CD_CNTRLLR
#define CD_CNTRLLR

#include <opencv2/highgui/highgui.hpp>
#include "colordetector.h"

class ColorDetectController {

  private:

   // the algorithm class
   ColorDetector *cdetect;
   
   cv::Mat image;   // The image to be processed
   cv::Mat result;  // The image result
   
	
  public:
	ColorDetectController() { // private constructor

		  //setting up the application
		  cdetect= new ColorDetector();
	}

	  // Sets the colour distance threshold
	  void setColorDistanceThreshold(int distance) {

		  cdetect->setColorDistanceThreshold(distance);
	  }

	  // Gets the colour distance threshold
	  int getColorDistanceThreshold() const {

		  return cdetect->getColorDistanceThreshold();
	  }

	  // Sets the colour to be detected
	  void setTargetColor(unsigned char red, unsigned char green, unsigned char blue) {

		  cdetect->setTargetColor(blue,green,red);
	  }

	  // Gets the colour to be detected
	  void getTargetColour(unsigned char &red, unsigned char &green, unsigned char &blue) const {

		  cv::Vec3b colour= cdetect->getTargetColor();

		  red= colour[2];
		  green= colour[1];
		  blue= colour[0];
	  }

	  // Sets the input image. Reads it from file.
	  bool setInputImage(std::string filename) {

		  image= cv::imread(filename);

		  return !image.empty();
	  }

	  // Returns the current input image.
	  const cv::Mat getInputImage() const {

		  return image;
	  }

	  // Performs image processing.
	  void process() {

		  result= cdetect->process(image);
	  }
	  

	  // Returns the image result from the latest processing.
	  const cv::Mat getLastResult() const {

		  return result;
	  }

	  // Deletes all processor objects created by the controller.
	  ~ColorDetectController() {

		  delete cdetect;
	  }
};

#endif
