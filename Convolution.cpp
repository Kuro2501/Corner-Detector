#include "Convolution.h"

vector<float> Convolution::GetKernel()
{
	return this->_kernel;
}

void Convolution::SetKernel(vector<float> kernel, int kWidth, int kHeight)
{
	this->_kernel = kernel;
	this->_kernelHeight = kHeight;
	this->_kernelWidth = kWidth;
}

int Convolution::DoConvolution(const Mat& sourceImage, Mat& destinationImage)
{
	if (sourceImage.empty()) 
		return 1;

	int width = sourceImage.cols;
	int height = sourceImage.rows;
	int nChannels = sourceImage.channels();
	destinationImage = Mat(sourceImage.rows, sourceImage.cols, CV_32FC1, Scalar(0));

	int	xStart = this->_kernelWidth / 2, yStart = this->_kernelHeight / 2;
	int xEnd = width - (this->_kernelWidth - 1), yEnd = height - (this->_kernelHeight - 1);
	int widthStep = sourceImage.cols;

	uchar* pSrc = sourceImage.data + sourceImage.cols * yStart + xStart;
	float* pDst = (float*)(destinationImage.data) + sourceImage.cols * yStart + xStart;

	// Create offset table
	vector<int> offsets;
	int widthLimit = this->_kernelWidth / 2;
	int heightLimit = this->_kernelHeight / 2;
	for (int y = -heightLimit; y <= heightLimit; y++)
		for (int x = -widthLimit; x <= widthLimit; x++)
			offsets.push_back(y * widthStep + x);

	// Calculate convolution
	int size = this->_kernelWidth * this->_kernelHeight;
	for (int y = yStart; y < yEnd; y++, pSrc += widthStep, pDst += widthStep) {
		const uchar* pSrcRow = pSrc;
		float* pDstRow = pDst;
		for (int x = xStart; x < xEnd; x++, pSrcRow++, pDstRow++)
		{
			float sum = 0;
			for (int u = -yStart; u <= yStart; u++)
				for (int i = 0; i < size; i++)
					sum += pSrcRow[offsets[i]] * _kernel[i];
			*pDstRow = sum;
		}
	}
	return 0; // if success
}

Convolution::Convolution() {}
Convolution::~Convolution() {}
