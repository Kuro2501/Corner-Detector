#include "CornerDetector.h"
#include "SIFTMatching.h"

int main(int argc, char* argv[]) {
	string option, src1, src2, alpha, threshold;
	CornerDetector cornerDetector;
	if (argc == 4) {
		option = argv[1];
		src1 = argv[2];
		threshold = argv[3];
		Mat sourceImage = imread(src1);
		if (option == "detectBlob") {
			cornerDetector.detectBlob(sourceImage, stoi(threshold));
			imshow("Source Image", sourceImage);
			waitKey(0);
		}
		if (option == "detectDOG") {
			cornerDetector.detectDOG(sourceImage, stoi(threshold));
			imshow("Source Image", sourceImage);
			waitKey(0);
		}
	}
	if (argc == 5) {
		option = argv[1];
		src1 = argv[2];
		Mat sourceImage1 = imread(src1);
		if (option == "detectHarrist") {
			alpha = argv[3];
			threshold = argv[4];
			cornerDetector.detectHarrist(sourceImage1, stof(alpha), stoi(threshold));
			imshow("Source Image", sourceImage1);
			waitKey(0);
		}
		if (option == "matchBySIFT") {
			src2 = argv[3];
			Mat sourceImage2 = imread(src2);
			threshold = argv[4];
			SIFTMatching s;
			Mat dst;
			s.matchBySIFT(sourceImage1, sourceImage2, dst, stoi(threshold));
		}
	}
}