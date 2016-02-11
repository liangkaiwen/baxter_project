#include "stdafx.h"
#include "opencvUtil.h"

using std::cout;
using std::endl;

cv::Rect scaleRectCentered(const cv::Rect& rect, float factor)
{
	cv::Rect result = rect;
	result.x += result.width * (1 - factor) * 0.5;
	result.y += result.height * (1 - factor) * 0.5;
	result.width *= factor;
	result.height *= factor;
	return result;
}

cv::Rect scaleRectWithImage(const cv::Rect& rect, float factor)
{
	cv::Rect result = rect;
	result.x *= factor;
	result.y *= factor;
	result.width *= factor;
	result.height *= factor;
	return result;
}

cv::Mat applyRectToMask(const cv::Mat& in, const cv::Rect& rect)
{
	cv::Mat rectMask = cv::Mat::zeros(in.size(), CV_8UC1);
	cv::Mat rectMaskToSet(rectMask, rect);
	rectMaskToSet.setTo(255);
	cv::Mat result;
	cv::bitwise_and(in, rectMask, result);
	return result;
}

void showInWindow(std::string name, cv::Mat image)
{
	if (image.empty()) return;

	int min_width = 150;
	cv::Mat image_to_show;
	if (image.cols < min_width) {
		image_to_show = cv::Mat(image.rows, min_width, image.type());
		image_to_show.setTo(cv::Scalar(0,0,0));
		image.copyTo(image_to_show(cv::Rect(0,0,image.cols,image.rows)));
	}
	else {
		image_to_show = image;
	}
	cv::namedWindow(name);
	cv::imshow(name, image_to_show);
}

cv::Mat createMxN(int m, int n, const std::vector<cv::Mat>& images, int height, int width)
{
	if (images.empty()) return cv::Mat();

	cv::Size first_size = images[0].size();
	int first_type = images[0].type();
	for (int i = 1; i < images.size(); ++i) {
		if (images[i].size() != first_size) {
			cout << "Size mismatch" << endl;
			return cv::Mat();
		}
		if (images[i].type() != first_type) {
			cout << "Type mismatch" << endl;
			return cv::Mat();
		}
	}

	int source_height = images[0].rows;
	int source_width = images[0].cols;
	int dest_height = std::max(height, source_height);
	int dest_width = std::max(width, source_width);

	cv::Mat result(dest_height * m, dest_width * n, images[0].type(), cv::Scalar::all(0));
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			int v_index = row * n + col;
			if (v_index >= images.size()) continue;
			if (images[v_index].data) {
				images[v_index].copyTo(result(cv::Rect(col * dest_width, row * dest_height, source_width, source_height)));
			}
		}
	}
	return result;
}

cv::Mat floatC1toCharC3(cv::Mat image)
{
	cv::Mat image_8UC1;
	image.convertTo(image_8UC1, CV_8UC1, 255);
	cv::Mat result;
	cv::cvtColor(image_8UC1, result, CV_GRAY2BGR);
	return result;
}

cv::Mat create1x2(const cv::Mat& image_0, const cv::Mat& image_1)
{
	std::vector<cv::Mat> image_v;
	image_v.push_back(image_0);
	image_v.push_back(image_1);
	return createMxN(1,2,image_v);
}