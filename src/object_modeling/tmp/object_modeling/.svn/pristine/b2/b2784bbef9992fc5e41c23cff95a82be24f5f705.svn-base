#include "frame.h"

#include "Noise.h"

#include "KernelSetFloat.h"
#include "KernelSetInt.h"

Frame::Frame(boost::shared_ptr<OpenCLAllKernels> all_kernels)
	: all_kernels_(all_kernels),
	image_buffer_color(all_kernels->getCL()),
	image_buffer_depth(all_kernels->getCL()),
	image_buffer_segments(all_kernels->getCL()),
	image_buffer_points(all_kernels->getCL()),
	image_buffer_normals(all_kernels->getCL()),
	image_buffer_add_depth_weights(all_kernels->getCL()),
	image_buffer_add_color_weights(all_kernels->getCL()),
	image_buffer_align_weights(all_kernels->getCL())
{
}


cv::Mat Frame::getPrettyDepth() const 
{
	cv::Mat depth_8;
	float display_factor = 255.f/5.f;
	mat_depth.convertTo(depth_8, CV_8U, display_factor);
	return depth_8;
}

cv::Mat Frame::getPNGDepth(float depth_factor) const 
{
	cv::Mat depth_16;
	mat_depth.convertTo(depth_16, CV_16U, depth_factor);
	return depth_16;
}

cv::Mat Frame::getMaskedColor() const 
{
	// look for valid depth
	cv::Mat mask = mat_depth > 0;
	cv::Mat result;
	mat_color_bgra.copyTo(result, mask);
	return result;
}

// You are slowly breaking this up into multiple functions as needed
void Frame::copyToImageBuffers() 
{
	KernelSetFloat _KernelSetFloat(*all_kernels_);

	// write color image as uchar4
	image_buffer_color.setMat(mat_color_bgra);

	// write depth image as float
	image_buffer_depth.setMat(mat_depth);

	copySegmentsToImageBuffer();

	// if no weights, set to 1's of appropriate size
	if (mat_add_depth_weights.empty()) {
		//mat_add_depth_weights = cv::Mat(mat_color.size(), CV_32F, cv::Scalar(1));
		image_buffer_add_depth_weights.resize(image_buffer_color.getRows(), image_buffer_color.getCols(), 1, CV_32F);
		_KernelSetFloat.runKernel(image_buffer_add_depth_weights.getBuffer(), image_buffer_add_depth_weights.getSizeElements(), 1);
	}
	else {
		image_buffer_add_depth_weights.setMat(mat_add_depth_weights);
	}

	copyAddColorWeightsToImageBuffer();

	if (mat_align_weights.empty()) {
		//mat_align_weights = cv::Mat(mat_color.size(), CV_32F, cv::Scalar(1));
		image_buffer_align_weights.resize(image_buffer_color.getRows(), image_buffer_color.getCols(), 1, CV_32F);
		_KernelSetFloat.runKernel(image_buffer_align_weights.getBuffer(), image_buffer_align_weights.getSizeElements(), 1);
	}
	else {
		image_buffer_align_weights.setMat(mat_align_weights);
	}
}

void Frame::copySegmentsToImageBuffer()
{
	KernelSetInt _KernelSetInt(*all_kernels_);
	if (mat_segments.empty()) {
		//mat_segments = cv::Mat(mat_color.size(), CV_32S, cv::Scalar(0));
		image_buffer_segments.resize(image_buffer_color.getRows(), image_buffer_color.getCols(), 1, CV_32S);
		_KernelSetInt.runKernel(image_buffer_segments.getBuffer(), image_buffer_segments.getSizeElements(), 0);
	}
	else {
		image_buffer_segments.setMat(mat_segments);
	}
}

void Frame::copyAddColorWeightsToImageBuffer()
{
	KernelSetFloat _KernelSetFloat(*all_kernels_);
	if (mat_add_color_weights.empty()) {
		//mat_add_color_weights = cv::Mat(mat_color.size(), CV_32F, cv::Scalar(1));
		image_buffer_add_color_weights.resize(image_buffer_color.getRows(), image_buffer_color.getCols(), 1, CV_32F);
		_KernelSetFloat.runKernel(image_buffer_add_color_weights.getBuffer(), image_buffer_add_color_weights.getSizeElements(), 1);
	}
	else {
		image_buffer_add_color_weights.setMat(mat_add_color_weights);
	}
}

void Frame::reduceToMaxDepth(float max_depth) 
{
	if (max_depth < 0) return;
	cv::Mat mask = mat_depth < max_depth;
	applyMaskToDepth(mask);
}

void Frame::circleMaskPixelRadius(int pixel_radius, int center_row, int center_col)
{
	if (pixel_radius < 0) return;
	for (int row = 0; row < mat_depth.rows; ++row) {
		for (int col = 0; col < mat_depth.cols; ++col) {
			int row_d = row - center_row;
			int col_d = col - center_col;
			if (row_d * row_d + col_d * col_d > pixel_radius * pixel_radius) {
				mat_depth.at<float>(row,col) = 0;
			}
		}
	}
}

void Frame::applyMaskToDepth(const cv::Mat & mask)
{
	cv::Mat fresh (mat_depth.size(), mat_depth.type(), cv::Scalar(0));
	mat_depth.copyTo(fresh, mask);
	mat_depth = fresh;
}

std::map<std::string, cv::Mat> Frame::setColorWeights(float max_edge_sigmas, float max_distance_transform)
{
	std::map<std::string, cv::Mat> result_debug_images;
	// in a dream world, this all on the GPU

	// first get a depth edge image
	// edges are those pixels with no depth, or those adjacent to a pixel more than 3 sigmas away

	// mask will be 1 for pixels with valid depth not adjacent to a depth boundary
	const int rows = mat_color_bgra.rows;
	const int cols = mat_color_bgra.cols;
	cv::Mat mat_edge_mask = (mat_depth > 0);
	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			uchar & mask = mat_edge_mask.at<uchar>(row, col);
			float const& d = mat_depth.at<float>(row, col);
			float max_depth_diff = max_edge_sigmas * Noise::simpleAxial(d);
			// I default all edges to 0 (image boundaries unreliable)

			if (row == 0 || fabs(d-mat_depth.at<float>(row-1,col)) > max_depth_diff) {
				mask = 0;
				continue;
			}
			if (row == rows - 1 || fabs(d-mat_depth.at<float>(row+1,col)) > max_depth_diff) {
				mask = 0;
				continue;
			}
			if (col == 0 || fabs(d-mat_depth.at<float>(row,col-1)) > max_depth_diff) {
				mask = 0;
				continue;
			}
			if (col == cols - 1 || fabs(d-mat_depth.at<float>(row,col+1)) > max_depth_diff) {
				mask = 0;
				continue;
			}
		}
	}

	// debug
	result_debug_images["mat_edge_mask"] = mat_edge_mask;

	// build the distance image
	cv::Mat mat_distance;
	cv::distanceTransform(mat_edge_mask, mat_distance, CV_DIST_L2, CV_DIST_MASK_PRECISE);

	// threshold to max
	//cv::threshold(mat_distance, mat_distance, max_distance_transform, max_distance_transform, cv::THRESH_TRUNC);
	cv::threshold(mat_distance, mat_distance, max_distance_transform, 1, cv::THRESH_BINARY);

	//mat_color_weights = mat_distance / max_distance_transform; // plus a little bit
	mat_add_color_weights = mat_distance + 1e-6;

	result_debug_images["mat_add_color_weights"] = mat_add_color_weights;

	return result_debug_images;
}