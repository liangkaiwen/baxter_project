#pragma once

#include "parameters.h"
#include "typedefs.h"

void learnWindowMouseCallback(int event, int x, int y, int flags, void* param);

class LearnHistogram
{
public:
	LearnHistogram(const Parameters& params);

	void reset();
	void init(cv::Mat initial_histogram);
	bool learn(const FrameT& frame);
	void showMarginalHistograms(const cv::Mat& histogram);
	void destroyWindows();
	cv::Mat getHistogram();

protected:
	const Parameters& params;
	std::set<std::string> windowsToDestroy;
	void showInWindowToDestroy(const std::string& name, const cv::Mat& image);
	cv::Mat histogram_sum;

};

