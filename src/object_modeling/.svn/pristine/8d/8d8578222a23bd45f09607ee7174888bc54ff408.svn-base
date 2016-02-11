#include "stdafx.h"

#include "OpenCLOptimize.h"

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

OpenCLOptimize::OpenCLOptimize(boost::shared_ptr<OpenCLAllKernels> all_kernels, int image_channel_count)
	: all_kernels_(all_kernels),
	frame_initialized(false),
	render_initialized(false),
	error_point_count(0),
	image_channel_count(image_channel_count),
	computeErrorAndGradientReducedKernel(all_kernels->getKernel("computeErrorAndGradientReduced")),
	reduceErrorAndGradientKernel(all_kernels->getKernel("reduceErrorAndGradient"))
{
}

void OpenCLOptimize::checkInitException() const
{
	if (!frame_initialized || !render_initialized) throw std::runtime_error ("OpenCLTSDF not initialized");
}

void OpenCLOptimize::prepareFrameBuffersWithBuffers(
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
	cl::Buffer const& buffer_frame_weights)
{
	// frame camera params
	camera_f_frame.s[0] = camera_f_frame_x;
	camera_f_frame.s[1] = camera_f_frame_y;
	camera_c_frame.s[0] = camera_c_frame_x;
	camera_c_frame.s[1] = camera_c_frame_y;
	camera_size_frame.s[0] = camera_size_frame_x;
	camera_size_frame.s[1] = camera_size_frame_y;

    try {
        cout << "SOMETHING STUPID IS HAPPENING" << endl;
        //cl::Memory::getInfo();
        cl_uint reference_count = 0;
        bufferFrameImage.getInfo(CL_MEM_REFERENCE_COUNT, &reference_count);
        cout << "bufferFrameImage reference count: " << reference_count << endl;
    }
    catch (cl::Error er) {
        cout << "SOMETHING STUPID" << endl;
        cout << "cl::Error: " << oclErrorString(er.err()) << endl;
        //cout << "(absorbing exception)" << endl;
        throw er;
    }

	bufferFramePoints = buffer_frame_points;
	bufferFrameNormals = buffer_frame_normals;
	bufferFrameImage = buffer_frame_image;
	bufferFrameGradientX = buffer_frame_gradient_x;
	bufferFrameGradientY = buffer_frame_gradient_y;
	bufferFrameWeights = buffer_frame_weights;

	frame_initialized = true;
}

void OpenCLOptimize::prepareRenderedAndErrorBuffersWithBuffers(
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
	cl::Buffer const& buffer_image)
{
	CL& cl = all_kernels_->getCL();

	// render camera params
	camera_f_render.s[0] = camera_f_render_x;
	camera_f_render.s[1] = camera_f_render_y;
	camera_c_render.s[0] = camera_c_render_x;
	camera_c_render.s[1] = camera_c_render_y;

	// what subset of full camera is actually rendered:
	render_rect.s[0] = render_rect_x;
	render_rect.s[1] = render_rect_y;
	render_rect.s[2] = render_rect_width;
	render_rect.s[3] = render_rect_height;

	error_point_count = render_rect_width * render_rect_height;

	bufferRenderedPoints = buffer_points;
	bufferRenderedNormals = buffer_normals;
	bufferRenderedImage = buffer_image;

	// This is still a bunch of buffer allocation!  Should use expanding buffers...

	// the error buffers depend on error_point_count
	// Note that with the full opencl reduction, these are not needed, but the bufferErrorVector should still be used for cdi
	size_t bufferErrorVectorSize = sizeof(float) * getErrorVectorSize();
	bufferErrorVector = cl::Buffer (cl.context, 0, bufferErrorVectorSize);
	size_t bufferErrorMatrixSize = sizeof(float) * getErrorMatrixSize();
	bufferErrorMatrix = cl::Buffer (cl.context, 0, bufferErrorMatrixSize);
	size_t bufferRobustWeightsSize = bufferErrorVectorSize;
	bufferRobustWeights = cl::Buffer (cl.context, 0, bufferRobustWeightsSize);

	// so do the reduced error vectors...
	size_t bufferReducedLHSSize = sizeof(float) * 21 * getNumBlocks(error_point_count, numThreads);
	bufferReducedLHS = cl::Buffer (cl.context, 0, bufferReducedLHSSize);
	size_t bufferReducedRHSSize = sizeof(float) * 6 * getNumBlocks(error_point_count, numThreads);
	bufferReducedRHS = cl::Buffer (cl.context, 0, bufferReducedRHSSize);

	// also the needed temporary reduce buffers
	// too big...
	bufferTempReduceLHS = cl::Buffer (cl.context, 0, bufferReducedLHSSize); 
	bufferTempReduceRHS = cl::Buffer (cl.context, 0, bufferReducedRHSSize);

	render_initialized = true;
}


void OpenCLOptimize::computeErrorAndGradient(
	float max_distance, float min_normal_dot_product,
	float weight_icp, float weight_color,
	bool huber_icp, bool huber_color,
	const Eigen::Affine3f& x_transform, 
	Eigen::Matrix<float,6,6> & LHS, Eigen::Matrix<float,6,1> & RHS, 
	float *error_vector, float *error_matrix, float *robust_weights)
{
	CL& cl = all_kernels_->getCL();

	checkInitException();

	try {
		cl_float16 cl_pose = getCLPose(x_transform);
		cl_int do_write_error = (error_vector != NULL);
		cl_int do_write_gradient = (error_matrix != NULL);
		cl_int do_write_robust_weights = (robust_weights != NULL);

		//////////////////////////
		int kernel_arg = 0;

		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, bufferRenderedPoints);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, bufferRenderedNormals);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, bufferRenderedImage);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, bufferFramePoints);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, bufferFrameNormals);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, bufferFrameImage);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, bufferFrameGradientX);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, bufferFrameGradientY);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, bufferFrameWeights);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, bufferReducedLHS);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, bufferReducedRHS);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, sizeof(float)*numThreads*21, NULL);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, sizeof(float)*numThreads*6, NULL);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, do_write_error);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, bufferErrorVector);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, do_write_gradient);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, bufferErrorMatrix);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, do_write_robust_weights);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, bufferRobustWeights);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, render_rect);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, camera_size_frame);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, cl_pose);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, (cl_float)max_distance);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, (cl_float)min_normal_dot_product);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, (cl_float)weight_icp);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, (cl_float)weight_color);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, (cl_int)image_channel_count);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, camera_f_render);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, camera_c_render);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, camera_f_frame);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, camera_c_frame);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, (cl_int)error_point_count);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, (cl_int)huber_icp);
		computeErrorAndGradientReducedKernel.setArg(kernel_arg++, (cl_int)huber_color);
		//////////////////////////////////

		//cl::NDRange global(error_point_count);
		cl::NDRange global(numThreads * getNumBlocks(error_point_count, numThreads));
		cl::NDRange local(numThreads);
		cl.queue.enqueueNDRangeKernel(computeErrorAndGradientReducedKernel, cl::NullRange, global, local);
		if (do_write_error) {
			cl.queue.enqueueReadBuffer(bufferErrorVector, CL_FALSE, 0, sizeof(float) * getErrorVectorSize(), (void*) error_vector);
		}
		if (do_write_gradient) {
			cl.queue.enqueueReadBuffer(bufferErrorMatrix, CL_FALSE, 0, sizeof(float) * getErrorMatrixSize(), (void*) error_matrix);
		}
		if (do_write_robust_weights) {
			cl.queue.enqueueReadBuffer(bufferRobustWeights, CL_FALSE, 0, sizeof(float) * getErrorVectorSize(), (void*) robust_weights);
		}

		// and now....must call the reduction
		finalSingleKernelReduction(bufferReducedLHS, bufferReducedRHS, getNumBlocks(error_point_count, numThreads), LHS, RHS);

		cl.queue.finish(); // for earlier reads
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		printf("computeErrorAndGradientReducedKernel\n");
		throw er;
	}
}


void OpenCLOptimize::finalSingleKernelReduction(cl::Buffer LHS_in, cl::Buffer RHS_in, int n, Eigen::Matrix<float,6,6> & LHS, Eigen::Matrix<float,6,1> & RHS)
{
	CL& cl = all_kernels_->getCL();

	int inputSize = n;
	int numBlocks = getNumBlocks(inputSize, numThreads);

	// now reduce down
	cl::Buffer *inputBufferLHS = &LHS_in;
	cl::Buffer *outputBufferLHS = &bufferTempReduceLHS;
	cl::Buffer *inputBufferRHS = &RHS_in;
	cl::Buffer *outputBufferRHS = &bufferTempReduceRHS;
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
void OpenCLOptimize::computeErrorAndGradientNew(
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

	// needed by old reduction to be big enough
	// also the needed temporary reduce buffers
	// too big (really only need to be sized for the first reduced size getNumBlocks())
	bufferTempReduceLHS = cl::Buffer (all_kernels_->getCL().context, 0, sizeof(float) * LHS_float_count);
	bufferTempReduceRHS = cl::Buffer (all_kernels_->getCL().context, 0, sizeof(float) * RHS_float_count);

	finalSingleKernelReduction(LHS_terms.getBuffer(), RHS_terms.getBuffer(), total_count, result_LHS, result_RHS);
}
