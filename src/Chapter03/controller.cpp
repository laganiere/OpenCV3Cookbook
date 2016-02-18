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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>

#include "colorDetectController.h"


int main()
{
	// Create the controller
    ColorDetectController controller;


	// To display the result
	cv::namedWindow("Image");
    
	// The following code simulate a user Interface
	// based on the use of a controller
	// Interaction with user is simply done
	// using key pressed
	std::cout << "q: to quit" << std::endl;
	std::cout << "f: to input a filename" << std::endl;
	std::cout << "t: to input target color values" << std::endl;
	std::cout << "c: to input color distance threshold" << std::endl;
	std::cout << "v: to view the different parameter values" << std::endl;
	std::cout << "r: to run" << std::endl;

    char key=' ';
	std::string filename;

	while ((key=getchar()) != 'q') {

		switch (key) {
				uchar r,g,b;
			case 'f':  // read an image
				std::cout << std::endl << "Filename? ";
				std::cin >> filename; 
				std::cout << std::endl;
				if (controller.setInputImage(filename)) 
					std::cout << "...image successfully opened" << std::endl;
				else
					std::cout << "...cannot find image: " << filename << std::endl;
			
				break;
			case 't':  // input target color
				int ir,ig,ib;
				std::cout << std::endl << "Target color? ";
				std:: cin >> ir >> ig >> ib;
				std::cout << std::endl;
				controller.setTargetColor(ir,ig,ib);
				break;
			case 'c':  // input threshold
				int th;
				std::cout << std::endl << "Color distance threshold? ";
				std:: cin >> th;
				std::cout << std::endl;
				controller.setColorDistanceThreshold(th);
				break;
			case 'v':  // view the parameters
				std::cout << std::endl << "Image name: " << filename << std::endl;
				controller.getTargetColour(r,g,b);
				std::cout << std::endl << "Target color: " 
											<< static_cast<int>(r) << "," 
											<< static_cast<int>(g) << "," 
											<< static_cast<int>(b) << std::endl;
				std::cout << std::endl << "Distance thresdhold: " << controller.getColorDistanceThreshold() << std::endl;
				std::cout << std::endl;
				break;
			case 'i':  // show input image
				cv::imshow("Image",controller.getInputImage()); 
                cv::waitKey(10); // for window to repaint
                break;
			case 'r':  // run color detection
				controller.process();
				cv::imshow("Image",controller.getLastResult()); 
                cv::waitKey(10); // for window to repaint
				break;
		}	
	}

	return 0;
}

