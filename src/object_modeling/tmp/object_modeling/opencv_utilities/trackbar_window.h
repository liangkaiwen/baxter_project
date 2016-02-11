#pragma once

#include <opencv2/opencv.hpp>

#include <string>

class TrackbarWindow
{
public:
    static void trackbarCallback(int event, void* param);

    TrackbarWindow(int max_value, std::string window_name = "TrackbarWindow");

	void setMat(const cv::Mat & mat);

    int getTrackbarValue();

	void destroyWindows();
	
protected:
	std::string window_name_;
	cv::Mat mat_;
    int trackbar_value_;
    int trackbar_max_;
};

