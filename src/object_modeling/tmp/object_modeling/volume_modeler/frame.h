#pragma once

#include <opencv2/opencv.hpp>

#include <boost/shared_ptr.hpp>

#include "ImageBuffer.h"
#include "keypoints.h"

#include "ros_timestamp.h"

#include "OpenCLAllKernels.h"

class Frame
{
public:
	Frame(boost::shared_ptr<OpenCLAllKernels> all_kernels);
	
	////////////// public members here:

	// timestamp of some sort?
	cv::Mat mat_color_bgra; // uchar4
	cv::Mat mat_depth; // float32
	cv::Mat mat_segments; // int32

	// This affects weights in the depth and color volume:
	cv::Mat mat_add_depth_weights; // float32

	// This affects weights in the color volume, but is overwritten if params_volume_modeler.set_color_weights = true (default)
	cv::Mat mat_add_color_weights; // float32

	// This affects the weight for alignment
	cv::Mat mat_align_weights; // float32

	// GPU equivalents
	ImageBuffer image_buffer_color; // uchar4
	ImageBuffer image_buffer_depth; // float32
	ImageBuffer image_buffer_segments; // int32

	// These are computed as needed
	ImageBuffer image_buffer_points; // todo: remove eventually
	ImageBuffer image_buffer_normals;

	// weights for adding frame to model
	ImageBuffer image_buffer_add_depth_weights;
	ImageBuffer image_buffer_add_color_weights;

	// alignment weights
	ImageBuffer image_buffer_align_weights;

	// computed if needed
	boost::shared_ptr<Keypoints> keypoints;

	ROSTimestamp ros_timestamp;


	/////////////// functions:

	cv::Mat getPrettyDepth() const;

	cv::Mat getPNGDepth(float depth_factor = 10000.0f) const;

	cv::Mat getMaskedColor() const;

	void copyToImageBuffers();

	void copySegmentsToImageBuffer();

	void copyAddColorWeightsToImageBuffer();

	void reduceToMaxDepth(float max_depth);

	void circleMaskPixelRadius(int pixel_radius, int center_row, int center_col);

	void applyMaskToDepth(const cv::Mat & mask);

	std::map<std::string, cv::Mat> setColorWeights(float max_edge_sigmas, float max_distance_transform);

protected:
	boost::shared_ptr<OpenCLAllKernels> all_kernels_;

};
