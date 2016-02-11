#pragma once

#include "cll.h"
#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"

#include <Eigen/Geometry>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;



class OpenCLOptimizeNew
{
public:
    struct OptimizeDebugImages {
        cv::Mat weighted_error;
        cv::Mat debug_code;
        cv::Mat weights_distance;
        cv::Mat weights_huber;
    };

	static const int numThreads = 128;// public for debugging?

	OpenCLOptimizeNew(boost::shared_ptr<OpenCLAllKernels> all_kernels);

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

	// functions

	// compute but don't copy to CPU:
	void computeErrorICPAndColor(const Eigen::Affine3f& x_transform);
	void computeDerivativeICPAndColor(const Eigen::Affine3f& x_transform);

	// utility for GPU dot products
	float dotProduct(cl::Buffer V1, cl::Buffer V2, int n);
	float reduceFloatVector(cl::Buffer input, int n);
	void finalSingleKernelReduction(cl::Buffer LHS_in, cl::Buffer RHS_in, int n, Eigen::Matrix<float,6,6> & LHS, Eigen::Matrix<float,6,1> & RHS);

	// members

	boost::shared_ptr<OpenCLAllKernels> all_kernels_;


#if 0
	cl::Buffer bufferErrorVector;
	cl::Buffer bufferErrorMatrix;
	cl::Buffer bufferRobustWeights;
	cl::Buffer bufferReducedLHS;
	cl::Buffer bufferReducedRHS;
	cl::Buffer bufferTempReduceLHS;
	cl::Buffer bufferTempReduceRHS;
#endif

	BufferWrapper buffer_wrapper_reduce_lhs_;
	BufferWrapper buffer_wrapper_reduce_rhs_;
};

