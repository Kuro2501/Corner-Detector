#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include "CornerDetector.h";
using namespace cv;
using namespace std;

class SIFTMatching {
public:
	SIFTMatching() {}
	void matchBySIFT(const Mat& img1, const Mat& img2, Mat& dst, int threshold) {
		cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create();
		std::vector<KeyPoint> keypoints1, keypoints2;
		Mat descriptors1, descriptors2;
		CornerDetector cornerDetector;
		cornerDetector.detectBlob(img1, threshold, keypoints1);
		cornerDetector.detectBlob(img2, threshold, keypoints2);
		detector->compute(img1, keypoints1, descriptors1);
		detector->compute(img2, keypoints2, descriptors2);
		//-- Step 2: Matching descriptor vectors with a FLANN based matcher
   // Since SURF is a floating-point descriptor NORM_L2 is used
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
		std::vector< std::vector<DMatch> > knn_matches;
		matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
		//-- Filter matches using the Lowe's ratio test
		const float ratio_thresh = 0.5f;
		std::vector<DMatch> good_matches;
		for (size_t i = 0; i < knn_matches.size(); i++)
		{
			if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
			{
				good_matches.push_back(knn_matches[i][0]);
			}
		}
		//-- Draw matches
		Mat img_matches;
		drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
			Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		//-- Show detected matches
		namedWindow("Good Matches", WINDOW_FREERATIO);
		imshow("Good Matches", img_matches);
		waitKey();
		dst = img_matches;
	}
};
