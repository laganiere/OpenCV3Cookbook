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

	cv::HOGDescriptor hog(cv::Size((image.cols/16)*16, (image.rows/16)*16), // size of the window
		cv::Size(8, 4),    // block size
		cv::Size(8, 4),    // block stride
		cv::Size(4, 4),    // cell size
		9);                // number of bins

	std::vector<float> descriptors;
//	hog.compute(image(cv::Rect(50, 50, 64, 128)), descriptors, cv::Size(64, 128));
	hog.compute(image, descriptors);

	std::vector<float> vec;
	cv::Mat im(8, 8, CV_8U, cv::Scalar(0));
	std::cout << descriptors.size() << " = " << 9*(image.rows/4) * (image.cols/4) <<std::endl;
	for (int i = 0; i < 9; i++) { std::cout << descriptors[i] << std::endl;
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

		std::cout << name << std::endl;
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

		std::cout << name << std::endl;
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


	std::vector<float> desc;
	// compute descriptor of first 8 positive samples
	for (int i = 0; i < 8; i++)
		hogDesc.compute(positives[i], descriptors);
	// compute descriptor of first 8 negative samples
	for (int i = 0; i < 8; i++)
		hogDesc.compute(negatives[i], descriptors);

	/*

	tr<ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::POLY);
	svm->setGamma(3);

	Mat trainData; // one row per feature
	Mat labels;
	Ptr<ml::TrainData> tData = ml::TrainData::create(trainData, ml::SampleTypes::ROW_SAMPLE, labels);
	svm->train(tData);
	// ...
	Mat query; // input, 1channel, 1 row (apply reshape(1,1) if nessecary)
	Mat res;   // output
	svm->predict(query, res);


	Size(0, 0), Size(0, 0), locations);
	//variables
	char FullFileName[100];
	char FirstFileName[100] = "./images/upperbody";
	char SaveHogDesFileName[100] = "Positive.xml";
	int FileNum = 96;

	vector< vector < float> > v_descriptorsValues;
	vector< vector < Point> > v_locations;


	for (int i = 0; i< FileNum; ++i)
	{
		sprintf_s(FullFileName, "%s%d.png", FirstFileName, i + 1);
		printf("%s\n", FullFileName);

		//read image file
		Mat img, img_gray;
		img = imread(FullFileName);

		//resizing
		resize(img, img, Size(64, 48)); //Size(64,48) ); //Size(32*2,16*2)); //Size(80,72) ); 
										//gray
		cvtColor(img, img_gray, CV_RGB2GRAY);

		//extract feature
		HOGDescriptor d(Size(32, 16), Size(8, 8), Size(4, 4), Size(4, 4), 9);
		vector< float> descriptorsValues;
		vector< Point> locations;
		d.compute(img_gray, descriptorsValues, Size(0, 0), Size(0, 0), locations);

		//printf("descriptor number =%d\n", descriptorsValues.size() );
		v_descriptorsValues.push_back(descriptorsValues);
		v_locations.push_back(locations);
		//show image
		imshow("origin", img);

*/
     cv::waitKey();
}

/*
Mat VisualizeHoG(Mat& origImg, vector<float>& descriptorValues)
{
Mat color_origImg;
cvtColor(origImg, color_origImg, CV_GRAY2RGB);

float zoomFac = 3;
Mat visu;
resize(color_origImg, visu, Size(color_origImg.cols*zoomFac, color_origImg.rows*zoomFac));

int blockSize       = 16;
int cellSize        = 8;
int gradientBinSize = 9;
float radRangeForOneBin = M_PI/(float)gradientBinSize; // dividing 180° into 9 bins, how large (in rad) is one bin?

// prepare data structure: 9 orientation / gradient strenghts for each cell
int cells_in_x_dir = 64 / cellSize;
int cells_in_y_dir = 128 / cellSize;
int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
float*** gradientStrengths = new float**[cells_in_y_dir];
int** cellUpdateCounter   = new int*[cells_in_y_dir];
for (int y=0; y<cells_in_y_dir; y++)
{
gradientStrengths[y] = new float*[cells_in_x_dir];
cellUpdateCounter[y] = new int[cells_in_x_dir];
for (int x=0; x<cells_in_x_dir; x++)
{
gradientStrengths[y][x] = new float[gradientBinSize];
cellUpdateCounter[y][x] = 0;

for (int bin=0; bin<gradientBinSize; bin++)
gradientStrengths[y][x][bin] = 0.0;
}
}

// nr of blocks = nr of cells - 1
// since there is a new block on each cell (overlapping blocks!) but the last one
int blocks_in_x_dir = cells_in_x_dir - 1;
int blocks_in_y_dir = cells_in_y_dir - 1;

// compute gradient strengths per cell
int descriptorDataIdx = 0;
int cellx = 0;
int celly = 0;

for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
{
for (int blocky=0; blocky<blocks_in_y_dir; blocky++)
{
// 4 cells per block ...
for (int cellNr=0; cellNr<4; cellNr++)
{
// compute corresponding cell nr
int cellx = blockx;
int celly = blocky;
if (cellNr==1) celly++;
if (cellNr==2) cellx++;
if (cellNr==3)
{
cellx++;
celly++;
}

for (int bin=0; bin<gradientBinSize; bin++)
{
float gradientStrength = descriptorValues[ descriptorDataIdx ];
descriptorDataIdx++;

gradientStrengths[celly][cellx][bin] += gradientStrength;

} // for (all bins)


// note: overlapping blocks lead to multiple updates of this sum!
// we therefore keep track how often a cell was updated,
// to compute average gradient strengths
cellUpdateCounter[celly][cellx]++;

} // for (all cells)


} // for (all block x pos)
} // for (all block y pos)


// compute average gradient strengths
for (int celly=0; celly<cells_in_y_dir; celly++)
{
for (int cellx=0; cellx<cells_in_x_dir; cellx++)
{

float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

// compute average gradient strenghts for each gradient bin direction
for (int bin=0; bin<gradientBinSize; bin++)
{
gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
}
}
}


cout << "descriptorDataIdx = " << descriptorDataIdx << endl;

// draw cells
for (int celly=0; celly<cells_in_y_dir; celly++)
{
for (int cellx=0; cellx<cells_in_x_dir; cellx++)
{
int drawX = cellx * cellSize;
int drawY = celly * cellSize;

int mx = drawX + cellSize/2;
int my = drawY + cellSize/2;

rectangle(visu, Point(drawX*zoomFac,drawY*zoomFac), Point((drawX+cellSize)*zoomFac,(drawY+cellSize)*zoomFac), CV_RGB(100,100,100), 1);

// draw in each cell all 9 gradient strengths
for (int bin=0; bin<gradientBinSize; bin++)
{
float currentGradStrength = gradientStrengths[celly][cellx][bin];

// no line to draw?
if (currentGradStrength==0)
continue;

float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;

float dirVecX = cos( currRad );
float dirVecY = sin( currRad );
float maxVecLen = cellSize/2;
float scale = 2.5; // just a visualization scale, to see the lines better

// compute line coordinates
float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

// draw gradient visualization
line(visu, Point(x1*zoomFac,y1*zoomFac), Point(x2*zoomFac,y2*zoomFac), CV_RGB(0,255,0), 1);

} // for (all bins)

} // for (cellx)
} // for (celly)


// don't forget to free memory allocated by helper data structures!
for (int y=0; y<cells_in_y_dir; y++)
{
for (int x=0; x<cells_in_x_dir; x++)
{
delete[] gradientStrengths[y][x];
}
delete[] gradientStrengths[y];
delete[] cellUpdateCounter[y];
}
delete[] gradientStrengths;
delete[] cellUpdateCounter;

return visu;

} // get_hogdescriptor_visu


Cv::Mat anger, disgust;
// Load the data into anger and disgust
...
// Make sure anger.cols == disgust.cols
// Combine your features from different classes into one big matrix
int numPostives = anger.rows, numNegatives = disgust.rows;
int numSamples = numPostives+numNegatives;
int featureSize = anger.cols;
cv::Mat data(numSamples, featureSize, CV_32FC1) // Assume your anger matrix is in float type
cv::Mat positiveData = data.rowRange(0, numPostives);
cv::Mat negativeData = data.rowRange(numPostives, numSamples);
anger.copyTo(positiveData);
disgust.copyTo(negativeData);
// Create label matrix according to the big feature matrix
cv::Mat labels(numSamples, 1, CV_32SC1);
labels.rowRange(0, numPositives).setTo(cv::Scalar_<int>(1));
labels.rowRange(numPositives, numSamples).setTo(cv::Scalar_<int>(-1));
// Finally, train your model
cv::SVM model;
model.train(data, labels, cv::Mat(), cv::Mat(), cv::SVMParams());
*/