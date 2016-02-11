#include "opencv_utilities.h"

// stupid visual studio for M_PI
#define _USE_MATH_DEFINES 
#include <math.h>

#include <algorithm>
#include <iostream>
using std::cout;
using std::endl;

cv::Mat createMxN(int m, int n, const std::vector<cv::Mat>& images, int height, int width)
{
	if (images.empty()) return cv::Mat();

	cv::Size first_size = images[0].size();
	int first_type = images[0].type();
	for (size_t i = 1; i < images.size(); ++i) {
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
			if (v_index >= (int)images.size()) continue;
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

cv::Mat floatC1toCharC4(cv::Mat image)
{
	cv::Mat image_8UC1;
	image.convertTo(image_8UC1, CV_8UC1, 255);
	cv::Mat result;
	cv::cvtColor(image_8UC1, result, CV_GRAY2BGRA);
	return result;
}

cv::Mat create1x2(const cv::Mat& image_0, const cv::Mat& image_1)
{
	std::vector<cv::Mat> image_v;
	image_v.push_back(image_0);
	image_v.push_back(image_1);
	return createMxN(1,2,image_v);
}

void imshowScale(const std::string & name, const cv::Mat & image)
{
	if (image.empty()) return;
	double min, max;
	cv::minMaxLoc(image, &min, &max);

	cout << "imshowScale min max " << name << ":" << min << " " << max << endl;

	cv::Mat to_show;
	if (max > min) to_show = (image - min) / (max - min);
	else to_show = image * 0;
	cv::imshow(name, to_show);
}

// just returns result before if false...utility function
cv::Mat getImageWithBorder(cv::Mat input, int border_width, cv::Scalar color)
{
    cv::Mat result;
    cv::copyMakeBorder(input, result, border_width, border_width, border_width, border_width, cv::BORDER_CONSTANT, color);
    return result;
}


void drawVerticalLine(cv::Mat input_and_result, float min_value, float max_value, float line_value, float line_height_fraction, int extra_width, cv::Scalar color)
{
    int col = (line_value - min_value) / (max_value - min_value) * input_and_result.cols + 0.5;
    int pixel_height = line_height_fraction * input_and_result.rows + 0.5;
    cv::rectangle(input_and_result, cv::Point(col - extra_width, input_and_result.rows), cv::Point(col + extra_width, input_and_result.rows - pixel_height), color, CV_FILLED);
}

void drawVerticalCenterLine(cv::Mat input_and_result, cv::Scalar color)
{
    drawVerticalLine(input_and_result, -1, 1, 0, 1, 0, color);
}


void drawHistogramOnImage(cv::Mat input_and_result, const std::vector<float> & values, cv::Scalar color)
{
    if (values.empty()) return;

    // get overall sum
    float normalize_factor = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        normalize_factor += values[i];
    }

    int height = input_and_result.rows;

    float width_per_bin = (float) input_and_result.cols / (float) values.size();

    for (size_t i = 0; i < values.size(); ++i) {
        int pixel_height = normalize_factor > 0 ? (float) height * values[i] / normalize_factor + 0.5 : 1;
        int pixel_left = i * width_per_bin + 0.5;
        int pixel_right = (i+1) * width_per_bin + 0.5 - 2;

        // lower left to upper right
        cv::rectangle(input_and_result, cv::Point(pixel_left, height), cv::Point(pixel_right, height - pixel_height), color, CV_FILLED);
    }
}


float gaussianPDF(float mean, float variance, float x)
{
    if (variance <= 0) return 0;
    float normalize_factor = 1.f / (sqrt(variance * 2 * M_PI));
    float diff = x - mean;
    return normalize_factor * exp( - diff * diff / (2 * variance));
}

void drawGaussianOnImage(cv::Mat input_and_result, float min_value, float max_value, float mean, float variance, int histogram_bins_for_matching_height, cv::Scalar color)
{
    if (variance > 0) {
        cv::Point last_point;
        for (int col = 0; col < input_and_result.cols; col++) {
            float x_value = min_value + (max_value - min_value) * ((float)col / (float) input_and_result.cols);
            float pdf = gaussianPDF(mean, variance, x_value);

#if 0
            // whatever...this is wrong but at least I can see it..
            // produces fixed-height gaussians:
            float fudge_factor = sqrt(variance);
            int pdf_in_pixels = fudge_factor * pdf * input_and_result.rows;
#endif


            // fucking doing this even though it's wrong for now...god damn it
#if 1
            // this was reasonable, but meaningless
            float height_factor = (max_value - min_value) / 2.f; // fraction our min and max are relative to -1 to 1
            float fudge_factor = 1./10;
            int pdf_in_pixels = fudge_factor * height_factor * pdf * input_and_result.rows;
#endif

#if 0
            // DON'T KNOW WHY THIS IS WRONG...OH WELL
            // a histogram bin of pixel area
            float supposed_mean_height = 1. / (max_value - min_value);  // fucking right!
            float desired_mean_height_in_pixels = (float) input_and_result.total() / (float) histogram_bins_for_matching_height;
            float supposed_linear_conversion = desired_mean_height_in_pixels / supposed_mean_height;
            int pdf_in_pixels = pdf * supposed_linear_conversion;
#endif



            cv::Point point(col, input_and_result.rows - pdf_in_pixels);
            if (col > 0) {
                cv::line(input_and_result, last_point, point, color);
            }
            last_point = point;
        }
    }

    // also short line at mean?
    drawVerticalLine(input_and_result, min_value, max_value, mean, 0.1, 0, color);
}
