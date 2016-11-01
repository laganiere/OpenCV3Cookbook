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
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

void drawHOG(std::vector<float> &hog, cv::Mat &image, float scale=1.0) {

	const float PI = 3.1415927;
	int numberOfBins = hog.size();
	float binStep = PI / numberOfBins;
	float maxLength = image.rows * 1.4142;
	float cx = image.cols / 2.;
	float cy = image.rows / 2.;

	for (int bin = 0; bin < numberOfBins; bin++) {

		float angle = bin*binStep;
		float dirX = cos(angle);
		float dirY = sin(angle);
		float length = 0.5*maxLength*hog[bin];
		int color = cv::saturate_cast<uchar>(hog[bin] * 255 * 3);

		float x1 = cx - dirX * length * scale;
		float y1 = cy - dirY * length * scale;
		float x2 = cx + dirX * length * scale;
		float y2 = cy + dirY * length * scale;

		cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), CV_RGB(color, color, color));
	}
}

void drawHOGDescriptors(const cv::Mat &image, cv::Mat &hogImage, cv::Size cellSize, int binSize) {

	cv::HOGDescriptor hog(cv::Size((image.cols / cellSize.width) * cellSize.width, 
		                           (image.rows / cellSize.height) * cellSize.height),
		cellSize,    // block size
		cellSize,    // block stride
		cellSize,    // cell size
		binSize);    // number of bins

	std::vector<float> descriptors;
	hog.compute(image, descriptors);

	hogImage.create(image.rows, image.cols, CV_8U);

	std::vector<float> vec;
	vec.push_back(0.6);
	vec.push_back(0.1);
	vec.push_back(0.1);
	vec.push_back(0.5);
	vec.push_back(0.1);
	vec.push_back(0.1);
	vec.push_back(0.3);
	vec.push_back(0.3);
	vec.push_back(0.1);

	for (int i = 0; i < image.rows / cellSize.height; i++) {
		for (int j = 0; j < image.cols / cellSize.width; j++) {

			hogImage(cv::Rect(j*cellSize.width, i*cellSize.height, cellSize.width, cellSize.height));
			drawHOG(vec, hogImage(cv::Rect(j*cellSize.width, i*cellSize.height, 
				                           cellSize.width, cellSize.height)), 1.);
		}
	}
}

int main()
{
	cv::Mat image = imread("girl.jpg", cv::IMREAD_GRAYSCALE);
	cv::imshow("Original image", image);

	cv::HOGDescriptor hog(cv::Size((image.cols / 16) * 16, (image.rows / 16) * 16), // size of the window
		cv::Size(8, 4),    // block size
		cv::Size(8, 4),    // block stride
		cv::Size(4, 4),    // cell size
		9);                // number of bins

	std::vector<float> descriptors;
	//	hog.compute(image(cv::Rect(50, 50, 64, 128)), descriptors, cv::Size(64, 128));
	hog.compute(image, descriptors);

	std::vector<float> vec;
	cv::Mat im(8, 8, CV_8U, cv::Scalar(0));
	std::cout << descriptors.size() << " = " << 9 * (image.rows / 4) * (image.cols / 4) << std::endl;
	for (int i = 0; i < 9; i++) {
		std::cout << descriptors[i] << std::endl;
		vec.push_back(descriptors[i]);
	}
	float sum = 0.;
	for (int i = 0; i < 9; i++) sum += descriptors[i];
	std::cout << sum << std::endl << std::endl;
	for (int i = 9; i < 18; i++) std::cout << descriptors[i] << std::endl;
	sum = 0.;
	for (int i = 9; i < 18; i++) sum += descriptors[i];
	std::cout << sum << std::endl << std::endl;

	vec.clear();
	vec.push_back(0.6);
	vec.push_back(0.1);
	vec.push_back(0.1);
	vec.push_back(0.1);
	vec.push_back(0.1);
	vec.push_back(0.1);
	vec.push_back(0.3);
	vec.push_back(0.3);
	vec.push_back(0.1);

	drawHOG(vec, im, 1.);
	cv::imshow("HOG", im);

	cv::Mat hogImage;
	drawHOGDescriptors(image, hogImage, cv::Size(64, 64), 9);
	cv::imshow("HOG image", hogImage);

	// generate the filename
	std::vector<std::string> imgs;
	std::string prefix = "stopSamples/stop";
	std::string ext = ".png";

	// loading the positive samples
	std::vector<cv::Mat> positives;

	for (long i = 0; i < 10; i++) {

		std::string name(prefix);
		std::ostringstream ss; ss << std::setfill('0') << std::setw(2) << i; name += ss.str();
		name += ext;

		positives.push_back(cv::imread(name, cv::IMREAD_GRAYSCALE));
	}

	// the first 8 positive samples
	cv::Mat posSamples(2 * positives[0].rows, 4 * positives[0].cols, CV_8U);
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 4; j++) {

			positives[i * 4 + j].copyTo(posSamples(cv::Rect(j*positives[i * 4 + j].cols, i*positives[i * 4 + j].rows, positives[i * 4 + j].cols, positives[i * 4 + j].rows)));

		}

	cv::imshow("Positive samples", posSamples);


	// loading the negative samples
	std::string nprefix = "stopSamples/neg";
	std::vector<cv::Mat> negatives;

	for (long i = 0; i < 10; i++) {

		std::string name(nprefix);
		std::ostringstream ss; ss << std::setfill('0') << std::setw(2) << i; name += ss.str();
		name += ext;

		negatives.push_back(cv::imread(name, cv::IMREAD_GRAYSCALE));
	}

	// the first 8 negative samples
	cv::Mat negSamples(2 * negatives[0].rows, 4 * negatives[0].cols, CV_8U);
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 4; j++) {

			negatives[i * 4 + j].copyTo(negSamples(cv::Rect(j*negatives[i * 4 + j].cols, i*negatives[i * 4 + j].rows, negatives[i * 4 + j].cols, negatives[i * 4 + j].rows)));
		}

	cv::imshow("Negative samples", negSamples);

	cv::HOGDescriptor hogDesc(positives[0].size(), // size of the window
		cv::Size(8, 8),    // block size
		cv::Size(4, 4),    // block stride
		cv::Size(4, 4),    // cell size
		9);                // number of bins

	// compute first descriptor 
	std::vector<float> desc;
	hogDesc.compute(positives[0], desc);

	// the matrix of sample descriptors
	int featureSize = desc.size();
	int numberOfSamples = positives.size() - 2    // we do not use the last 2 samples
		+ negatives.size() - 2;
	cv::Mat samples(numberOfSamples, featureSize, CV_32FC1);

	// fill first row with first descriptor
	for (int i = 0; i < featureSize; i++)
		samples.ptr<float>(0)[i] = desc[i];

	// compute descriptor of first 8 positive samples
	for (int j = 1; j < 8; j++) {
		hogDesc.compute(positives[j], desc);
		// fill the next row with current descriptor
		for (int i = 0; i < featureSize; i++)
			samples.ptr<float>(j)[i] = desc[i];
	}

	// compute descriptor of first 8 negative samples
	for (int j = 0; j < 8; j++) {
		hogDesc.compute(negatives[j], desc);
		// fill the next row with current descriptor
		for (int i = 0; i < featureSize; i++)
			samples.ptr<float>(j + 8)[i] = desc[i];
	}

	// Create label matrix according to the big feature matrix
	cv::Mat labels(numberOfSamples, 1, CV_32SC1);
	labels.rowRange(0, 8) = 1.0;   // labels of positive samples
	labels.rowRange(8, 16) = -1.0; // labels of negative samples


	// create SVM classifier
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::LINEAR);

	// prepare the training data
	cv::Ptr<cv::ml::TrainData> trainingData =
		cv::ml::TrainData::create(samples, cv::ml::SampleTypes::ROW_SAMPLE, labels);

	// SVM training
	svm->train(trainingData);

	cv::Mat queries(4, featureSize, CV_32FC1);

	// fill the rows with query descriptors
	hogDesc.compute(positives[8], desc);
	for (int i = 0; i < featureSize; i++)
		queries.ptr<float>(0)[i] = desc[i];
	hogDesc.compute(positives[9], desc);
	for (int i = 0; i < featureSize; i++)
		queries.ptr<float>(1)[i] = desc[i];
	hogDesc.compute(negatives[8], desc);
	for (int i = 0; i < featureSize; i++)
		queries.ptr<float>(2)[i] = desc[i];
	hogDesc.compute(negatives[9], desc);
	for (int i = 0; i < featureSize; i++)
		queries.ptr<float>(3)[i] = desc[i];
	cv::Mat predictions;

	// Test the classifier with the last two pos and neg samples
	svm->predict(queries, predictions);
	std::cout << "Predicted classes:\n";
	
	for (int i = 0; i < 4; i++)
		std::cout << "query: " << i << ": " << ((predictions.at<float>(i, 0) < 0.0)? "Negative" : "Positive") << std::endl;

	cv::waitKey();
}