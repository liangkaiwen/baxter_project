#pragma once

#include <opencv2/opencv.hpp>

#include "params_mask_object.h"


void learnWindowMouseCallback(int event, int x, int y, int flags, void* param);

class LearnHistogram
{
public:
	LearnHistogram(const ParamsMaskObject & params_mask_object);

	void reset();

	void init(const cv::Mat & initial_histogram);
	
	bool learn(const cv::Mat & image_color);
	
	void showMarginalHistograms(const cv::Mat& histogram);
	
	void destroyWindows();
	
	cv::Mat getHistogram();

protected:
	void showInWindowToDestroy(const std::string & name, const cv::Mat & image);

	// members
	ParamsMaskObject params_mask_object_;
	std::set<std::string> windows_to_destroy_;
	cv::Mat histogram_sum_;
};

