#pragma once
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Convolution.h"
#include <iostream>
using namespace cv;
using namespace std;

class CornerDetector
{
public:
	
	Mat detectHarrist(const Mat& sourceImage, float alpha, int thresholds) {
		Mat srcGray, dstImage;
		cvtColor(sourceImage, srcGray, COLOR_BGR2GRAY);
		GaussianBlur(srcGray, srcGray, Size(3, 3), 1.0, 1.0, BORDER_DEFAULT);
		dstImage = sourceImage.clone();

		Convolution convolution;
		vector<float> kernelX = { -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0 };
		vector<float> kernelY = { -1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0 };
		
		// Find gradient X, gradient Y
		Mat gx, gy;

		int width = srcGray.cols;
		int height = srcGray.rows;
		int nChannels = srcGray.channels();
		int widthStep = srcGray.step[0];

		convolution.SetKernel(kernelX, 3, 3);
		convolution.DoConvolution(srcGray, gx);
		convolution.SetKernel(kernelY, 3, 3);
		convolution.DoConvolution(srcGray, gy);

		Mat Ix2, Iy2, Ixy;
		pow(gx, 2.0, Ix2);
		pow(gy, 2.0, Iy2);
		multiply(gx, gy, Ixy);

		// Gaussian blur 
		Mat Ix2_blured, Iy2_blured, Ixy_blured;
		GaussianBlur(Ix2, Ix2_blured, Size(7, 7), 2.0, 0.0, BORDER_DEFAULT);
		GaussianBlur(Iy2, Iy2_blured, Size(7, 7), 0.0, 2.0, BORDER_DEFAULT);
		GaussianBlur(Ixy, Ixy_blured, Size(7, 7), 2.0, 2.0, BORDER_DEFAULT);
		
		// Compute R = det(M) − k(trace(M))2
		Mat Ix2Iy2, IxyIxy;
		multiply(Ix2_blured, Iy2_blured, Ix2Iy2);
		multiply(Ixy_blured, Ixy_blured, IxyIxy);

		Mat R, traceM;
		pow((Ix2_blured + Iy2_blured), 2.0, traceM);
		R = (Ix2Iy2 - IxyIxy) - alpha * traceM;
		
		// Compare with Thresholds
		Mat dst_norm, dst_norm_scaled;
		normalize(R, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
		convertScaleAbs(dst_norm, dst_norm_scaled);
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if ((int)dst_norm.at<float>(y, x) > thresholds) {
					circle(sourceImage, Point(x, y), 1, Scalar(0, 0, 255), 1, 8, 0);
				}
			}
		}

		return dstImage;
	}

	Mat detectBlob(const Mat& sourceImage, int threshold) {
		Mat srcGray, dstImage;
		cvtColor(sourceImage, srcGray, COLOR_BGR2GRAY);
		GaussianBlur(srcGray, srcGray, Size(11, 11), 1.0, 1.0, BORDER_DEFAULT);
		dstImage = sourceImage.clone();

		int width = srcGray.cols;
		int height = srcGray.rows;
		int nChannels = srcGray.channels();
		int widthStep = srcGray.step[0];

		// Laplace
		Laplacian(srcGray, srcGray, CV_8UC1, 3, 1, 0, BORDER_DEFAULT);

		// Compare with Thresholds
		Mat dst_norm;
		normalize(srcGray, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if ((int)dst_norm.at<float>(y, x) > threshold) {
					circle(sourceImage, Point(x, y), 12, Scalar(0, 0, 255), 1, 8, 0);
				}
			}
		}

		return dstImage;
	}

	Mat detectDOG(const Mat& sourceImage, int threshold) {
		Mat srcGray, dstImage;
		cvtColor(sourceImage, srcGray, COLOR_BGR2GRAY);
		dstImage = sourceImage.clone();

		int width = srcGray.cols;
		int height = srcGray.rows;
		int nChannels = srcGray.channels();
		int widthStep = srcGray.step[0];

		// Calculate low_sigma Gaussian blur, high_sigma Gaussian blur and DoG
		Mat low_sigma, high_sigma, DoG;
		//GaussianBlur(srcGray, low_sigma, Size(5, 5), 1.0, 1.0, BORDER_DEFAULT);
		//GaussianBlur(srcGray, high_sigma, Size(7, 7), 1.0, 1.0, BORDER_DEFAULT);
		Laplacian(srcGray, low_sigma, CV_8UC1, 5, 1, 0 , BORDER_DEFAULT);
		Laplacian(srcGray, high_sigma, CV_8UC1, 7, 1, 0, BORDER_DEFAULT);
		DoG = low_sigma - high_sigma;

		// Compare with Thresholds
		Mat dst_norm;
		normalize(DoG, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if ((int)dst_norm.at<float>(y, x) > threshold) {
					circle(sourceImage, Point(x, y), 12, Scalar(0, 0, 255), 1, 8, 0);
				}
			}
		}

		return dstImage;
	}

	Mat detectBlob(const Mat& sourceImage, int threshold, vector<KeyPoint>& keypoints) {
		Mat srcGray, dstImage;
		cvtColor(sourceImage, srcGray, COLOR_BGR2GRAY);
		GaussianBlur(srcGray, srcGray, Size(5, 5), 1.0, 1.0, BORDER_DEFAULT);
		dstImage = sourceImage.clone();

		int width = srcGray.cols;
		int height = srcGray.rows;
		int nChannels = srcGray.channels();
		int widthStep = srcGray.step[0];

		// Laplace
		Laplacian(srcGray, srcGray, CV_8UC1, 3, 1, 0, BORDER_DEFAULT);

		// Compare with Thresholds
		Mat dst_norm;
		normalize(srcGray, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
		vector<Point2f> key;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if ((int)dst_norm.at<float>(y, x) > threshold) {
					key.push_back(Point2f(x * 1.f, y * 1.f));
				}
			}
		}
		for (int i = 0; i < key.size(); i++) {
			keypoints.push_back(KeyPoint(key[i], 1.f));
		}

		return dstImage;
	}

	CornerDetector() {};
	~CornerDetector() {};
};