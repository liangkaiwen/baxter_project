#pragma once

#include "cll.h"
#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"

#include <Eigen/Geometry>
#include <boost/thread.hpp>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

class OpenCLOptimize
{
public:
    struct OptimizeDebugImages {
        cv::Mat weighted_error;
        cv::Mat debug_code;
        cv::Mat weights_distance;
        cv::Mat weights_huber;
    };

	static const int numThreads = 128;// public for debugging?

	OpenCLOptimize(boost::shared_ptr<OpenCLAllKernels> all_kernels, int image_channel_count);

	void prepareFrameBuffersWithBuffers(
		float camera_f_frame_x,
		float camera_f_frame_y,
		float camera_c_frame_x,
		float camera_c_frame_y,
		int camera_size_frame_x,
		int camera_size_frame_y,
		cl::Buffer const& buffer_frame_points,
		cl::Buffer const& buffer_frame_normals,
		cl::Buffer const& buffer_frame_image,
		cl::Buffer const& buffer_frame_gradient_x,
		cl::Buffer const& buffer_frame_gradient_y,
		cl::Buffer const& buffer_frame_weights);

	void prepareRenderedAndErrorBuffersWithBuffers(
		float camera_f_render_x,
		float camera_f_render_y,
		float camera_c_render_x,
		float camera_c_render_y,
		int render_rect_x,
		int render_rect_y,
		int render_rect_width,
		int render_rect_height,
		cl::Buffer const& buffer_points,
		cl::Buffer const& buffer_normals,
		cl::Buffer const& buffer_image);

	void computeErrorAndGradient(
		float max_distance, float min_normal_dot_product,
		float weight_icp, float weight_color,
		bool huber_icp, bool huber_color,
		const Eigen::Affine3f& x_transform, 
		Eigen::Matrix<float,6,6> & LHS, Eigen::Matrix<float,6,1> & RHS, 
		float *error_vector, float *error_matrix, float *robust_weights);

	size_t getErrorPointCount() {return error_point_count;}
	size_t getErrorVectorSize() {return error_point_count * (1 + image_channel_count);}
	size_t getErrorMatrixSize() {return error_point_count * (1 + image_channel_count) * 6;}

	void computeErrorAndGradientNew(
		const ImageBuffer & ib_render_points,
		const ImageBuffer & ib_render_normals,
		const ImageBuffer & ib_render_image,
		const ImageBuffer & ib_frame_points,
		const ImageBuffer & ib_frame_normals,
		const ImageBuffer & ib_frame_image,
		const ImageBuffer & ib_frame_image_gradient_x,
		const ImageBuffer & ib_frame_image_gradient_y,
		const Eigen::Array2f & camera_focal_render,
		const Eigen::Array2f & camera_center_render,
		const Eigen::Array2f & camera_focal_frame,
		const Eigen::Array2f & camera_center_frame,
		const Eigen::Affine3f& x_transform, 
		const float outlier_max_distance,
		const float outlier_min_normal_dot_product,
		const float huber_param_icp,
		const float huber_param_image,
		const float weight_icp,
		const float weight_image,
		Eigen::Matrix<float,6,6> & result_LHS,
		Eigen::Matrix<float,6,1> & result_RHS,
		const bool generate_debug_images,
		OptimizeDebugImages & result_debug_images_icp,
		OptimizeDebugImages & result_debug_images_image
		);

protected:
	void checkInitException() const;

	// compute but don't copy to CPU:
	void computeErrorICPAndColor(const Eigen::Affine3f& x_transform);
	void computeDerivativeICPAndColor(const Eigen::Affine3f& x_transform);

	// utility for GPU dot products
	float dotProduct(cl::Buffer V1, cl::Buffer V2, int n);
	float reduceFloatVector(cl::Buffer input, int n);
	void finalSingleKernelReduction(cl::Buffer LHS_in, cl::Buffer RHS_in, int n, Eigen::Matrix<float,6,6> & LHS, Eigen::Matrix<float,6,1> & RHS);

	boost::shared_ptr<OpenCLAllKernels> all_kernels_;
	bool frame_initialized;
	bool render_initialized;

	////// kernels
	cl::Kernel computeErrorAndGradientReducedKernel;
	cl::Kernel reduceErrorAndGradientKernel;

	//////// buffers
	cl_float2 camera_f_frame;
	cl_float2 camera_c_frame;
	cl_int2 camera_size_frame;
	cl::Buffer bufferFramePoints;
	cl::Buffer bufferFrameNormals;
	cl::Buffer bufferFrameImage;
	cl::Buffer bufferFrameGradientX;
	cl::Buffer bufferFrameGradientY;
	cl::Buffer bufferFrameWeights;

	cl_float2 camera_f_render;
	cl_float2 camera_c_render;
	cl_int4 render_rect;
	cl::Buffer bufferRenderedPoints;
	cl::Buffer bufferRenderedNormals;
	cl::Buffer bufferRenderedImage;

	size_t error_point_count;
	size_t image_channel_count;

	cl::Buffer bufferErrorVector;
	cl::Buffer bufferErrorMatrix;
	cl::Buffer bufferRobustWeights;
	cl::Buffer bufferReducedLHS;
	cl::Buffer bufferReducedRHS;
	cl::Buffer bufferTempReduceLHS;
	cl::Buffer bufferTempReduceRHS;
};

