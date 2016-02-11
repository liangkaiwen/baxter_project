#pragma once

#include <string>
#include <opencv2/opencv.hpp>

struct HistogramConstants {
public:
	static const float hist_hranges[];
	static const float hist_sranges[];
	static const float hist_vranges[];
	static const float* hist_hsv_ranges[];
	static const int hist_hsv_channels[];
};

// channel select
cv::Mat view3DHistogram1D(const cv::Mat& histogram, int channel);
// channel marginalize
cv::Mat view3DHistogram2D(const cv::Mat& histogram, int channel);

bool saveHistogram(const std::string& filename, const cv::Mat& histogram);
bool loadHistogram(const std::string& filename, cv::Mat& histogram);