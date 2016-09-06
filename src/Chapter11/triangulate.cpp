/*------------------------------------------------------------------------------------------*\
This file contains material supporting chapter 11 of the book:
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


#include "triangulate.h"

cv::Vec3d triangulate(const cv::Mat &p1, const cv::Mat &p2, const cv::Vec2d &u1, const cv::Vec2d &u2) {

	// triangulate using Linear LS-Method
	
	cv::Matx43d A(u1(0)*p1.at<double>(2, 0) - p1.at<double>(0, 0), u1(0)*p1.at<double>(2, 1) - p1.at<double>(0, 1), u1(0)*p1.at<double>(2, 2) - p1.at<double>(0, 2),
		u1(1)*p1.at<double>(2, 0) - p1.at<double>(1, 0), u1(1)*p1.at<double>(2, 1) - p1.at<double>(1, 1), u1(1)*p1.at<double>(2, 2) - p1.at<double>(1, 2),
		u2(0)*p2.at<double>(2, 0) - p2.at<double>(0, 0), u2(0)*p2.at<double>(2, 1) - p2.at<double>(0, 1), u2(0)*p2.at<double>(2, 2) - p2.at<double>(0, 2),
		u2(1)*p2.at<double>(2, 0) - p2.at<double>(1, 0), u2(1)*p2.at<double>(2, 1) - p2.at<double>(1, 1), u2(1)*p2.at<double>(2, 2) - p2.at<double>(1, 2));

	cv::Matx41d B(p1.at<double>(0, 3) - u1(0)*p1.at<double>(2, 3),
		          p1.at<double>(1, 3) - u1(1)*p1.at<double>(2, 3),
		          p2.at<double>(0, 3) - u2(0)*p2.at<double>(2, 3),
		          p2.at<double>(1, 3) - u2(1)*p2.at<double>(2, 3));
	cv::Vec3d X;// (3, 1, CV_64F);

	cv::solve(A, B, X, cv::DECOMP_SVD);

	return X;
}

void triangulate(const cv::Mat &p1, const cv::Mat &p2, const std::vector<cv::Vec2d> &pts1, const std::vector<cv::Vec2d> &pts2, std::vector<cv::Vec3d> &pts3D) {

	for (int i = 0; i < pts1.size(); i++) {

		pts3D.push_back(triangulate(p1, p2, pts1[i], pts2[i]));
	}
}
