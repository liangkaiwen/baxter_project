#include "trackbar_window.h"

#include <iostream>
using std::cout;
using std::endl;

void TrackbarWindow::trackbarCallback(int event, void* param)
{
}

TrackbarWindow::TrackbarWindow(int max_value, std::string window_name)
	: window_name_(window_name),
      trackbar_value_(0),
      trackbar_max_(max_value)
{
}

void TrackbarWindow::setMat(const cv::Mat & mat)
{
	cv::namedWindow(window_name_);
    cv::createTrackbar("trackbar", window_name_, &trackbar_value_, trackbar_max_, TrackbarWindow::trackbarCallback);
	cv::imshow(window_name_, mat);
}

int TrackbarWindow::getTrackbarValue()
{
    return trackbar_value_;
}

void TrackbarWindow::destroyWindows()
{
	cv::destroyWindow(window_name_);
}

