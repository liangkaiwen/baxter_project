#pragma once

#include <opencv2/opencv.hpp>

#include <string>

class PickPixel
{
public:
	static void mouseCallback(int event, int x, int y, int flags, void* param);

	PickPixel(std::string window_name = "PickPixel");

	void setMat(const cv::Mat & mat);

	cv::Point2i getPixel();

	void destroyWindows();
	
protected:
	std::string window_name_;
	cv::Mat mat_;
	cv::Point2i pixel_;
};

