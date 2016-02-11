#include "stdafx.h"
#include "histogramUtil.h"

using std::cout;
using std::endl;

const float HistogramConstants::hist_hranges[] = { 0, 180 };
const float HistogramConstants::hist_sranges[] = { 0, 256 };
const float HistogramConstants::hist_vranges[] = { 0, 256 };
const float* HistogramConstants::hist_hsv_ranges[] = { HistogramConstants::hist_hranges, HistogramConstants::hist_sranges, HistogramConstants::hist_vranges };
const int HistogramConstants::hist_hsv_channels[] = {0, 1, 2};

cv::Mat view3DHistogram1D(const cv::Mat& histogram, int channel)
{
	if (histogram.dims != 3) throw std::runtime_error ("histogram.dims != 3");
	if (histogram.type() != CV_32F) throw std::runtime_error ("(histogram.type() != CV_32F)");

	// marginalize
	int size = histogram.size[channel];
	cv::Mat marginal_histogram(1, &size, CV_32FC1, cv::Scalar::all(0));

	for (int i = 0; i < histogram.size[0]; i++) {
		for (int j = 0; j < histogram.size[1]; j++) {
			for (int k = 0; k < histogram.size[2]; k++) {
				float to_add = histogram.at<float>(i, j, k);
				if (channel == 0) marginal_histogram.at<float>(i) += to_add;
				else if (channel == 1) marginal_histogram.at<float>(j) += to_add;
				else if (channel == 2) marginal_histogram.at<float>(k) += to_add;
				else throw std::runtime_error("bad channel");
			}
		}
	}

	// create image
	double maxVal=0;
	minMaxLoc(marginal_histogram, 0, &maxVal, 0, 0);
	int scale = 10;
	cv::Mat histImg = cv::Mat::zeros(scale, size * scale, CV_8UC3);
	for (int i = 0; i < size; i++) {
		float binVal = marginal_histogram.at<float>(i);
		int intensity = cvRound(binVal*255/maxVal);
		rectangle( histImg, cv::Point(i*scale, 0),
			cv::Point( (i+1)*scale - 1, scale - 1),
			cv::Scalar::all(intensity),
			CV_FILLED );
	}

	return histImg;
}

cv::Mat view3DHistogram2D(const cv::Mat& histogram, int channel)
{
	if (histogram.dims != 3) throw std::runtime_error ("histogram.dims != 3");
	if (histogram.type() != CV_32F) throw std::runtime_error ("(histogram.type() != CV_32F)");

	int channel_1, channel_2;

	if (channel == 0) {
		channel_1 = 1;
		channel_2 = 2;
	}
	else if (channel == 1) {
		channel_1 = 0;
		channel_2 = 2;
	}
	else if (channel == 2) {
		channel_1 = 0;
		channel_2 = 1;
	}
	else throw std::runtime_error ("bad channel");

	// marginalize
	int sizes[] = {histogram.size[channel_1], histogram.size[channel_2]};
	cv::Mat marginal_histogram(2, sizes, CV_32FC1, cv::Scalar::all(0));

	for (int i = 0; i < histogram.size[0]; i++) {
		for (int j = 0; j < histogram.size[1]; j++) {
			for (int k = 0; k < histogram.size[2]; k++) {
				float to_add = histogram.at<float>(i, j, k);
				if (channel == 0) marginal_histogram.at<float>(j, k) += to_add;
				else if (channel == 1) marginal_histogram.at<float>(i, k) += to_add;
				else if (channel == 2) marginal_histogram.at<float>(i, j) += to_add;
				else throw std::runtime_error("bad channel");
			}
		}
	}

	// create image
	double maxVal=0;
	minMaxLoc(marginal_histogram, 0, &maxVal, 0, 0);
	int scale = 10;
	cv::Mat histImg = cv::Mat::zeros(sizes[1] * scale, sizes[0] * scale, CV_8UC3);
	for (int i = 0; i < sizes[0]; i++) {
		for (int j = 0; j < sizes[1]; j++) {
			float binVal = marginal_histogram.at<float>(i, j);
			int intensity = cvRound(binVal*255/maxVal);
			rectangle( histImg, cv::Point(i * scale, j * scale),
				cv::Point( (i+1)*scale - 1, (j+1)*scale - 1),
				cv::Scalar::all(intensity),
				CV_FILLED );
		}
	}

	return histImg;
}

bool saveHistogram(const std::string& filename, const cv::Mat& histogram)
{
	cv::FileStorage fsw(filename, cv::FileStorage::WRITE);
	if (!fsw.isOpened()) {
		cout << "Failed to save histogram: " << filename << endl;
		return false;
	}
	fsw << "histogram" << histogram;
	fsw.release();
	return true;
}

bool loadHistogram(const std::string& filename, cv::Mat& histogram)
{
	cv::FileStorage fsr(filename, cv::FileStorage::READ);
	if (!fsr.isOpened()) {
		cout << "Failed to open histogram: " << filename << endl;
		return false;
	}
	fsr["histogram"] >> histogram;
	fsr.release();
	return true;
}