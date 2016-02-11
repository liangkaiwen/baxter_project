#pragma once

#include "typedefs.h"
#include "parameters.h"
#include "runningStatistics.h"
#include "cloudToImage.hpp"
#include "opencvUtil.h"

template <typename Scalar>
class ICPCombinedFunctor
{
public:
	// boilerplate for EigenLM functor:
	typedef Scalar Scalar;
	enum {
		InputsAtCompileTime = Eigen::Dynamic,
		ValuesAtCompileTime = Eigen::Dynamic
	};
	typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
	typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
	typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

	// members:
	// could use friend of ObjectModeler to skip some of this
	const Parameters& params;
	const G2OStereoProjector<typename KeypointPointT, typename KeypointPointT>& projector_;
	const Eigen::Affine3f& current_pose_;
	const FrameT& frame_;
	OpenCLOptimize *opencl_optimize_;
	OpenCLImages *opencl_images_;

	// set in initRender
	bool initRenderCalled_;
	Eigen::Vector2f render_proj_f_;
	Eigen::Vector2f render_proj_c_;
	cv::Rect render_rect_;
	// set in OLD initRender
	CloudICPTargetT::ConstPtr rendered_cloud_with_normals_;
	cv::Mat rendered_image_for_dense_color_;

	// set in initFrame
	int initialized_octave_;

	// created in constructor:
	// the number of channels for these 3 should match:
	std::vector<cv::Mat> frame_image_for_dense_color_vec_;
	std::vector<cv::Mat> frame_image_gradient_x_vec_;
	std::vector<cv::Mat> frame_image_gradient_y_vec_;
	std::vector<cv::Rect> frame_object_rect_vec_;
	std::vector<cv::Mat> frame_object_mask_vec_;

	// these will be 0 if negative, value if positive
	float weight_icp_;
	float weight_color_;
	// set this once in constructor
	float min_normal_dot_;

	// updated each operator()
	mutable float last_error_icp_;
	mutable float last_error_icp_max_;
	mutable float last_error_color_;
	mutable float last_error_color_max_;
	mutable float last_error_total_;
	mutable int last_error_icp_count_;
	mutable int last_error_color_count_;

	// maintain timing stats
	mutable RunningStatistics rs_f;
	mutable RunningStatistics rs_df;

	// save debug images
	mutable std::vector<cv::Mat> combined_debug_image_v;

	ICPCombinedFunctor(	const Parameters& params,
		const G2OStereoProjector<KeypointPointT, KeypointPointT>& projector,
		const Eigen::Affine3f& current_pose,
		const FrameT& frame,
		OpenCLOptimize* opencl_optimize,
		OpenCLImages* opencl_images);

	cv::Mat extractImageChannels(cv::Mat image_rgb_8uc3);
	int getImageChannelCount() const;
	int getErrorChannelCount() const;
	void initRender(const Eigen::Vector2f &render_proj_f, const Eigen::Vector2f &render_proj_c, const cv::Rect& render_rect, const CloudICPTargetT::ConstPtr& rendered_cloud_with_normals);
	void initRender(const Eigen::Vector2f &render_proj_f, const Eigen::Vector2f &render_proj_c, const cv::Rect& render_rect, const RenderBuffers& render_buffers); // new
	void initFrame(int octave);
	void initOpenCLFrame(int octave);
	void initOpenCLRender();
	void initOpenCLRender(const RenderBuffers& render_buffers); // new
	Eigen::Matrix4f xToMatrix4f(const InputType& x) const;
	Eigen::Affine3f xToAffine3f(const InputType& x) const;
	int errorPoints() const;
	int valuesICPAndColor() const;
	int values() const;
	void combinedDebugImages(Eigen::Affine3f const& transform, std::vector<float> const& error_vector, bool new_ordering) const;
	template <typename Derived>	void errorICPAndColorOpenCL (const InputType &x, Eigen::MatrixBase<Derived> const &partial_fvec) const;
	template <typename Derived> void dfICPAndColorOpenCL(const InputType &x, Eigen::MatrixBase<Derived> const &partial_jmat) const;
	int operator() (const InputType &x, ValueType &fvec) const;
	int df (const InputType &x, JacobianType &jmat) const;
	void solveGaussNewton(const InputType &x, InputType &x_result, int& iterations, ValueType& error_vector);
	void solveGaussNewtonGPUFull(const InputType &x, InputType &x_result, int& iterations);
};

//#include "ICPCombinedFunctor.hpp"
//extern template class ICPCombinedFunctor<float>;
