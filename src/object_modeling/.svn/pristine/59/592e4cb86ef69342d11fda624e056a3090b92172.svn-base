#pragma once

#include <opencv2/opencv.hpp>

cv::Mat createMxN(int m, int n, const std::vector<cv::Mat>& images, int height = 0, int width = 0);

cv::Mat create1x2(const cv::Mat& image_0, const cv::Mat& image_1); // specialization

cv::Mat floatC1toCharC3(cv::Mat image);

cv::Mat floatC1toCharC4(cv::Mat image);

void imshowScale(const std::string & name, const cv::Mat & image);

cv::Mat getImageWithBorder(cv::Mat input, int border_width = 1, cv::Scalar color = cv::Scalar::all(0));

void drawVerticalLine(cv::Mat input_and_result, float min_value, float max_value, float line_value, float line_height_fraction, int extra_width, cv::Scalar color);

void drawVerticalCenterLine(cv::Mat input_and_result, cv::Scalar color = cv::Scalar(0,0,255));

void drawHistogramOnImage(cv::Mat input_and_result, const std::vector<float> & values, cv::Scalar color);

float gaussianPDF(float mean, float variance, float x);

void drawGaussianOnImage(cv::Mat input_and_result, float min_value, float max_value, float mean, float variance, int histogram_bins_for_matching_height, cv::Scalar color = cv::Scalar::all(255));
