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
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>


int main()
{
	cv::Mat image = imread("girl.jpg", cv::IMREAD_GRAYSCALE);
	cv::imshow("Original image", image);

	cv::HOGDescriptor hog(cv::Size(64, 128), // size of the window
		cv::Size(2, 2),    // block size
		cv::Size(2, 2),    // block stride
		cv::Size(8, 8),    // cell size
		9);                // number of bins

	std::vector<float> descriptors;
	hog.compute(image, descriptors);

	std::cout << descriptors.size();
	/*
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
*/