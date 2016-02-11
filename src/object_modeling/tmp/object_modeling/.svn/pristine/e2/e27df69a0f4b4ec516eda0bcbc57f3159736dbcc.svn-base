#include "stdafx.h" // why still this, lazy Peter

#include "OpenCLOptimizeNew.h"

#include <stdexcept>
#include <fstream>
#include <iostream>
using std::cout;
using std::endl;
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include <boost/assign.hpp>

#include "KernelOptimizeErrorAndJacobianICP.h"
#include "KernelOptimizeErrorAndJacobianImage.h"
#include "KernelOptimizeNormalEquationTerms.h"
#include "KernelSetFloat.h"
#include "KernelSetInt.h"

OpenCLOptimizeNew::OpenCLOptimizeNew(boost::shared_ptr<OpenCLAllKernels> all_kernels)
	: all_kernels_(all_kernels),
	buffer_wrapper_reduce_lhs_(all_kernels_->getCL()),
	buffer_wrapper_reduce_rhs_(all_kernels_->getCL())
{
}


void OpenCLOptimizeNew::finalSingleKernelReduction(cl::Buffer LHS_in, cl::Buffer RHS_in, int n, Eigen::Matrix<float,6,6> & LHS, Eigen::Matrix<float,6,1> & RHS)
{
	// todo: make Kernel...file 
	cl::Kernel reduceErrorAndGradientKernel(all_kernels_->getKernel("reduceErrorAndGradient"));

	CL& cl = all_kernels_->getCL();

	int inputSize = n;
	int numBlocks = getNumBlocks(inputSize, numThreads);

	// make temp buffers big enough
	// big enough is just based on numblocks (get reduced first time)
	buffer_wrapper_reduce_lhs_.reallocateIfNeeded(numBlocks * sizeof(float) * 21);
	buffer_wrapper_reduce_rhs_.reallocateIfNeeded(numBlocks * sizeof(float) * 6);

	// now reduce down
	cl::Buffer *inputBufferLHS = &LHS_in;
	cl::Buffer *outputBufferLHS = &buffer_wrapper_reduce_lhs_.getBuffer();
	cl::Buffer *inputBufferRHS = &RHS_in;
	cl::Buffer *outputBufferRHS = &buffer_wrapper_reduce_rhs_.getBuffer();
	while (inputSize >= numThreads) {
		int kernel_arg = 0;
		reduceErrorAndGradientKernel.setArg(kernel_arg++, *inputBufferLHS);
		reduceErrorAndGradientKernel.setArg(kernel_arg++, *inputBufferRHS);
		reduceErrorAndGradientKernel.setArg(kernel_arg++, *outputBufferLHS);
		reduceErrorAndGradientKernel.setArg(kernel_arg++, *outputBufferRHS);
		reduceErrorAndGradientKernel.setArg(kernel_arg++, sizeof(float)*numThreads*21, NULL);
		reduceErrorAndGradientKernel.setArg(kernel_arg++, sizeof(float)*numThreads*6, NULL);
		reduceErrorAndGradientKernel.setArg(kernel_arg++, inputSize);
		cl::NDRange global(numThreads * numBlocks);
		cl::NDRange local(numThreads);
		cl.queue.enqueueNDRangeKernel(reduceErrorAndGradientKernel, cl::NullRange, global, local);

		// prepare for next iteration
		cl::Buffer *swapBufferLHS = outputBufferLHS;
		outputBufferLHS = inputBufferLHS;
		inputBufferLHS = swapBufferLHS;
		cl::Buffer *swapBufferRHS = outputBufferRHS;
		outputBufferRHS = inputBufferRHS;
		inputBufferRHS = swapBufferRHS;

		inputSize = numBlocks;
		numBlocks = getNumBlocks(inputSize, numThreads);
	}

	// names for sanity
	outputBufferLHS = inputBufferLHS;
	outputBufferRHS = inputBufferRHS;
	size_t outputSize = inputSize;

	// get the remaining inputSize elements from the ->>>> inputBuffer (was outputBuffer)
	size_t LHS_float_count = 21 * outputSize;
	size_t RHS_float_count = 6 * outputSize;
	std::vector<float> reduceOutputLHS(LHS_float_count);
	std::vector<float> reduceOutputRHS(RHS_float_count);
	cl.queue.enqueueReadBuffer(*outputBufferLHS, CL_FALSE, 0, sizeof(float)*LHS_float_count, (void*) reduceOutputLHS.data());
	cl.queue.enqueueReadBuffer(*outputBufferRHS, CL_FALSE, 0, sizeof(float)*RHS_float_count, (void*) reduceOutputRHS.data());
	cl.queue.finish(); // needed or else true in last one

	// see if any zeros leak...sigh
	LHS = Eigen::Matrix<float,6,6>::Zero();
	RHS = Eigen::Matrix<float,6,1>::Zero();

	////////
	// first final reduce, THEN into the matrices
	std::vector<float> finalLHS(21, 0);
	for (int a = 0; a < 21; ++a) {
		for (int b = 0; b < outputSize; ++b) {
			finalLHS[a] += reduceOutputLHS[a*outputSize + b];
		}
	}
	std::vector<float> finalRHS(6, 0);
	for (int a = 0; a < 6; ++a) {
		for (int b = 0; b < outputSize; ++b) {
			finalRHS[a] += reduceOutputRHS[a*outputSize + b];
		}
	}

	size_t gpu_index = 0;
	for (int row = 0; row < 6; ++row) {
		for (int col = row; col < 6; ++col) {
			float value = finalLHS[gpu_index++];
			LHS(row,col) = value;
			LHS(col,row) = value;
		}
	}
	for (int row = 0; row < 6; ++row) {
		RHS(row) = finalRHS[row];
	}
}


// don't need to prepare this one
// todo: maybe put this in new file (once I write the "new" reduce to go with it)
// right now only a single image (no RGB, etc....just intensity in 10 cities)
void OpenCLOptimizeNew::computeErrorAndGradientNew(
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
	)
{
	KernelOptimizeErrorAndJacobianICP _KernelOptimizeErrorAndJacobianICP(*all_kernels_);
	KernelOptimizeErrorAndJacobianImage _KernelOptimizeErrorAndJacobianImage(*all_kernels_);
	KernelOptimizeNormalEquationTerms _KernelOptimizeNormalEquationTerms(*all_kernels_);

	KernelSetFloat _KernelSetFloat(*all_kernels_);
	KernelSetInt _KernelSetInt(*all_kernels_);

	// todo: allocate these once (but always set to zero)
	// note that multiple sizes will be used for multiscale
	// so consider a resize if needed (sigh)
	const int rows = ib_render_points.getRows();
	const int cols = ib_render_points.getCols();
	const int point_count = rows * cols;

	////////////////////
	// TODO: want to avoid reallocating every call
	// I think these need to be sized in multiple of threadcount
	const int number_of_images = 2; // icp + image

	BufferWrapper LHS_terms(all_kernels_->getCL());
	BufferWrapper RHS_terms(all_kernels_->getCL());
	const int total_count = numThreads * getNumBlocks(number_of_images * point_count, numThreads);
	const size_t LHS_float_count = 21 * total_count;
	const size_t RHS_float_count = 6 * total_count;
	LHS_terms.reallocateIfNeeded(LHS_float_count * sizeof(float));
	RHS_terms.reallocateIfNeeded(RHS_float_count * sizeof(float));

	// don't really need to set all to 0, just the "overflow" points
	_KernelSetFloat.runKernel(LHS_terms.getBuffer(), LHS_float_count, 0);
	_KernelSetFloat.runKernel(RHS_terms.getBuffer(), RHS_float_count, 0);


	//////////////////////
	// allocate local buffers
	// again, don't want to do this every time
	ImageBuffer weighted_error(all_kernels_->getCL());
	ImageBuffer weighted_jacobian(all_kernels_->getCL());
	ImageBuffer debug_code(all_kernels_->getCL());
	ImageBuffer debug_huber_weight(all_kernels_->getCL());
	ImageBuffer debug_distance_weight(all_kernels_->getCL());

	weighted_error.resize(rows, cols, 1, CV_32F);
	weighted_jacobian.resize(rows, cols, 6, CV_32F);
	debug_code.resize(rows, cols, 1, CV_32S);
	debug_huber_weight.resize(rows, cols, 1, CV_32F);
	debug_distance_weight.resize(rows, cols, 1, CV_32F);

	// put the icp terms in
	{
		_KernelSetFloat.runKernel(weighted_error.getBuffer(), point_count, 0);
		_KernelSetFloat.runKernel(weighted_jacobian.getBuffer(), 6 * point_count, 0);
		_KernelSetInt.runKernel(debug_code.getBuffer(), point_count, 0);
		_KernelSetFloat.runKernel(debug_huber_weight.getBuffer(), point_count, 0);
		_KernelSetFloat.runKernel(debug_distance_weight.getBuffer(), point_count, 0);

		// todo: use generate_debug_images to avoid extra kernel writes
		_KernelOptimizeErrorAndJacobianICP.runKernel(
			ib_render_points, ib_render_normals,
			ib_frame_points, ib_frame_normals,
			weighted_error, weighted_jacobian,
			debug_code, debug_huber_weight, debug_distance_weight,
			camera_focal_render, camera_center_render,
			camera_focal_frame, camera_center_frame,
			x_transform,
			outlier_max_distance,
			outlier_min_normal_dot_product,
			huber_param_icp);

		if (generate_debug_images) {
			result_debug_images_icp.weighted_error = weighted_error.getMat();
			result_debug_images_icp.debug_code = debug_code.getMat();
			result_debug_images_icp.weights_huber = debug_huber_weight.getMat();
			result_debug_images_icp.weights_distance = debug_distance_weight.getMat();
		}


		const int offset = 0;
		_KernelOptimizeNormalEquationTerms.runKernel(
			weighted_error, weighted_jacobian,
			LHS_terms, RHS_terms,
			total_count, offset, weight_icp);
	}

	// put image terms in
	{
		_KernelSetFloat.runKernel(weighted_error.getBuffer(), point_count, 0);
		_KernelSetFloat.runKernel(weighted_jacobian.getBuffer(), 6 * point_count, 0);
		_KernelSetInt.runKernel(debug_code.getBuffer(), point_count, 0);
		_KernelSetFloat.runKernel(debug_huber_weight.getBuffer(), point_count, 0);
		_KernelSetFloat.runKernel(debug_distance_weight.getBuffer(), point_count, 0);

		_KernelOptimizeErrorAndJacobianImage.runKernel(
			ib_render_points, ib_render_normals, ib_render_image,
			ib_frame_points, ib_frame_normals, ib_frame_image, ib_frame_image_gradient_x, ib_frame_image_gradient_y,
			weighted_error, weighted_jacobian,
			debug_code, debug_huber_weight, debug_distance_weight,
			camera_focal_render, camera_center_render,
			camera_focal_frame, camera_center_frame,
			x_transform,
			outlier_max_distance,
			outlier_min_normal_dot_product,
			huber_param_icp,
			huber_param_image);

		if (generate_debug_images) {
			result_debug_images_image.weighted_error = weighted_error.getMat();
			result_debug_images_image.debug_code = debug_code.getMat();
			result_debug_images_image.weights_huber = debug_huber_weight.getMat();
			result_debug_images_image.weights_distance = debug_distance_weight.getMat();
		}

		const int offset = point_count;
		_KernelOptimizeNormalEquationTerms.runKernel(
			weighted_error, weighted_jacobian,
			LHS_terms, RHS_terms,
			total_count, offset, weight_image);
	}



	//////////////////
	// call the old reduction
	// todo: update this

	finalSingleKernelReduction(LHS_terms.getBuffer(), RHS_terms.getBuffer(), total_count, result_LHS, result_RHS);
}
