#pragma once

#include <opencv2/opencv.hpp>

cv::Rect scaleRectCentered(const cv::Rect& rect, float factor);
cv::Rect scaleRectWithImage(const cv::Rect& rect, float factor);
cv::Mat applyRectToMask(const cv::Mat& in, const cv::Rect& rect);
void showInWindow(std::string name, cv::Mat image);
cv::Mat createMxN(int m, int n, const std::vector<cv::Mat>& images, int height = 0, int width = 0);
cv::Mat create1x2(const cv::Mat& image_0, const cv::Mat& image_1); // specialization
cv::Mat floatC1toCharC3(cv::Mat image);
