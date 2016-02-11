#pragma once

#include "ICPCombinedFunctor.h"

#include "pclUtil.hpp"

template <typename Scalar>
ICPCombinedFunctor<Scalar>::ICPCombinedFunctor(	const Parameters& params,
	const G2OStereoProjector<KeypointPointT, KeypointPointT>& projector,
	const Eigen::Affine3f& current_pose,
	const FrameT& frame,
	OpenCLOptimize* opencl_optimize,
	OpenCLImages* opencl_images) 
	: params(params),
	projector_(projector),
	current_pose_(current_pose),
	frame_(frame),
	opencl_optimize_(opencl_optimize),
	opencl_images_(opencl_images),
	initialized_octave_(-1),
	initRenderCalled_(false)
{
	// param adjustments:
	weight_icp_ = params.combined_weight_icp_points > 0 ? params.combined_weight_icp_points : 0;
	weight_color_ = params.combined_weight_color > 0 ? params.combined_weight_color : 0;
	min_normal_dot_ = cos(params.icp_normals_angle * M_PI / 180.0);

	last_error_icp_ = 0;
	last_error_icp_max_ = 0;
	last_error_color_ = 0;
	last_error_color_max_ = 0;
	last_error_total_ = 0;

	cv::Mat frame_image_full_size = extractImageChannels(frame_.image_color);
	if (params.color_blur_size > 0 && !params.color_blur_after_pyramid) {
		cv::GaussianBlur(frame_image_full_size, frame_image_full_size, cv::Size(params.color_blur_size, params.color_blur_size), 0);
	}

	frame_image_for_dense_color_vec_.push_back(frame_image_full_size);
	frame_object_rect_vec_.push_back(frame_.object_rect);
	frame_object_mask_vec_.push_back(frame.object_mask);

	// construct pyramid
	for (int i = 1; i < params.combined_octaves; i++) {
		cv::Mat downsampled;
		cv::pyrDown(frame_image_for_dense_color_vec_.back(), downsampled);

		// scale this?
		if (params.color_blur_size > 0 && params.color_blur_after_pyramid) {
			cv::GaussianBlur(downsampled, downsampled, cv::Size(params.color_blur_size, params.color_blur_size), 0);
		}

		frame_image_for_dense_color_vec_.push_back(downsampled);

		// also rect
		frame_object_rect_vec_.push_back(scaleRectWithImage(frame_object_rect_vec_.back(), 0.5));
		// and mask (for display only)
		cv::Mat smaller_mask;
		cv::resize(frame_object_mask_vec_.back(), smaller_mask, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
		frame_object_mask_vec_.push_back(smaller_mask);
	}

	// compute gradients for all!
	for (int i = 0; i < frame_image_for_dense_color_vec_.size(); i++) {
		frame_image_gradient_x_vec_.push_back(cv::Mat());
		frame_image_gradient_y_vec_.push_back(cv::Mat());
		cv::Sobel(frame_image_for_dense_color_vec_[i], frame_image_gradient_x_vec_.back(), -1, 1, 0, 3, 1.0/8.0);
		cv::Sobel(frame_image_for_dense_color_vec_[i], frame_image_gradient_y_vec_.back(), -1, 0, 1, 3, 1.0/8.0);
	}
}

template <typename Scalar>
cv::Mat ICPCombinedFunctor<Scalar>::extractImageChannels(cv::Mat image_rgb_8uc3)
{
	cv::Mat image_float;
	image_rgb_8uc3.convertTo(image_float, CV_32F, 1./255.);
	cv::Mat image_float_ycrcb;
	cv::cvtColor(image_float, image_float_ycrcb, CV_BGR2YCrCb);

	if (params.combined_image_error == Parameters::IMAGE_ERROR_YCBCR) {
		return image_float_ycrcb;
	}
	else if (params.combined_image_error == Parameters::IMAGE_ERROR_CBCR) {
		cv::Mat result = cv::Mat(image_float_ycrcb.size(), CV_32FC2);
		int from_to[] = {1,0, 2,1};
		cv::mixChannels(&image_float_ycrcb, 1, &result, 1, from_to, 2);
		return result;
	}
	else if (params.combined_image_error == Parameters::IMAGE_ERROR_Y) {
		cv::Mat result = cv::Mat(image_float_ycrcb.size(), CV_32FC1);
		int from_to[] = {0,0};
		cv::mixChannels(&image_float_ycrcb, 1, &result, 1, from_to, 1);
		return result;
	}
	else {
		throw new exception("Unknown image error");
	}
}

template <typename Scalar>
int ICPCombinedFunctor<Scalar>::getImageChannelCount() const
{
	if (params.combined_image_error == Parameters::IMAGE_ERROR_YCBCR) {
		return 3;
	}
	else if (params.combined_image_error == Parameters::IMAGE_ERROR_CBCR) {
		return 2;
	}
	else if (params.combined_image_error == Parameters::IMAGE_ERROR_Y) {
		return 1;
	}
	else {
		throw new exception("Unknown image error");
	}
}

template <typename Scalar>
int ICPCombinedFunctor<Scalar>::getErrorChannelCount() const
{
	return (1 + getImageChannelCount());
}

template <typename Scalar>
void ICPCombinedFunctor<Scalar>::initRender(const Eigen::Vector2f &render_proj_f, const Eigen::Vector2f &render_proj_c, const cv::Rect& render_rect, const CloudICPTargetT::ConstPtr& rendered_cloud_with_normals)
{
	render_proj_f_ = render_proj_f;
	render_proj_c_ = render_proj_c;
	render_rect_ = render_rect;
	rendered_cloud_with_normals_ = rendered_cloud_with_normals;
	cv::Mat rendered_image_bgr = cloudToImage(*rendered_cloud_with_normals_);
	rendered_image_for_dense_color_ = extractImageChannels(rendered_image_bgr);

	initOpenCLRender();

	initRenderCalled_ = true;
}

template <typename Scalar>
void ICPCombinedFunctor<Scalar>::initRender(const Eigen::Vector2f &render_proj_f, const Eigen::Vector2f &render_proj_c, const cv::Rect& render_rect, const RenderBuffers& render_buffers)
{
	render_proj_f_ = render_proj_f;
	render_proj_c_ = render_proj_c;
	render_rect_ = render_rect;

	initOpenCLRender(render_buffers);

	initRenderCalled_ = true;
}


template <typename Scalar>
void ICPCombinedFunctor<Scalar>::initFrame(int octave)
{
	if (octave < 0 || octave >= params.combined_octaves) throw new exception ("octave < 0 || octave >= params.combined_octaves");

	initOpenCLFrame(octave);

	initialized_octave_ = octave;
}

template <typename Scalar>
void ICPCombinedFunctor<Scalar>::initOpenCLFrame(int octave)
{
	if (octave < 0 || octave >= params.combined_octaves) throw new exception ("octave < 0 || octave >= params.combined_octaves");

	// prepare the frame points and normals
	// Just do nearest neighbor for points (for now)
	int rows_image = frame_image_for_dense_color_vec_[octave].rows;
	int cols_image = frame_image_for_dense_color_vec_[octave].cols;

	size_t frame_points_size = rows_image * cols_image;
	std::vector<float> frame_points (frame_points_size * 4, std::numeric_limits<float>::quiet_NaN());
	std::vector<float> frame_normals (frame_points_size * 4, std::numeric_limits<float>::quiet_NaN());
	int scale = 1 << octave;

	cv::Rect object_rect_scaled;
	object_rect_scaled.x = frame_.object_rect.x / scale;
	object_rect_scaled.y = frame_.object_rect.y / scale;
	object_rect_scaled.width = frame_.object_rect.width / scale;
	object_rect_scaled.height = frame_.object_rect.height / scale;

	// only setting points in object_rect
	for (int row_object = 0; row_object * scale < frame_.object_cloud_ptr->height; ++row_object) {
		for (int col_object = 0; col_object * scale < frame_.object_cloud_ptr->width; ++col_object) {
			const PointT& p = frame_.object_cloud_ptr->at(col_object * scale, row_object * scale);
			const pcl::Normal& n = frame_.object_normal_cloud_ptr->at(col_object * scale, row_object * scale);
			int row_image = row_object + object_rect_scaled.y;
			int col_image = col_object + object_rect_scaled.x;
			int image_vector_index = (row_image * cols_image + col_image) * 4;
			frame_points[image_vector_index] = p.x;
			frame_points[image_vector_index+1] = p.y;
			frame_points[image_vector_index+2] = p.z;
			frame_points[image_vector_index+3] = 1;
			frame_normals[image_vector_index] = n.normal_x;
			frame_normals[image_vector_index+1] = n.normal_y;
			frame_normals[image_vector_index+2] = n.normal_z;
			frame_normals[image_vector_index+3] = 0;
		}
	}

	Eigen::Vector2f f_scaled = projector_.getFocalLengths() / scale;
	Eigen::Vector2f c_scaled = projector_.getCenters() / scale;

	// weights based on frame image
	cv::Mat frame_weights_trivial(frame_image_for_dense_color_vec_[octave].size(), CV_32FC1, cv::Scalar::all(1.0));

	opencl_optimize_->prepareFrameBuffers(
		f_scaled[0], f_scaled[1], c_scaled[0], c_scaled[1],
		object_rect_scaled.x, object_rect_scaled.y, object_rect_scaled.width, object_rect_scaled.height,
		cols_image, rows_image, 
		frame_points.data(), frame_normals.data(), (float*) frame_image_for_dense_color_vec_[octave].data, (float*) frame_weights_trivial.data, (float*) frame_image_gradient_x_vec_[octave].data, (float*) frame_image_gradient_y_vec_[octave].data);

}

template <typename Scalar>
void ICPCombinedFunctor<Scalar>::initOpenCLRender()
{
	// prepare the rendered points
	int rendered_points_size = rendered_cloud_with_normals_->width * rendered_cloud_with_normals_->height;
	std::vector<float> rendered_points (rendered_points_size * 4, std::numeric_limits<float>::quiet_NaN());
	std::vector<float> rendered_normals (rendered_points_size * 4, std::numeric_limits<float>::quiet_NaN());
	for (int row = 0; row < rendered_cloud_with_normals_->height; row++) {
		for (int col = 0; col < rendered_cloud_with_normals_->width; col++) {
			const PointICPTargetT& p = rendered_cloud_with_normals_->at(col, row);
			int rendered_vector_index = row * rendered_cloud_with_normals_->width + col;
			rendered_points[4*rendered_vector_index] = p.x;
			rendered_points[4*rendered_vector_index+1] = p.y;
			rendered_points[4*rendered_vector_index+2] = p.z;
			rendered_points[4*rendered_vector_index+3] = 1;
			rendered_normals[4*rendered_vector_index] = p.normal_x;
			rendered_normals[4*rendered_vector_index+1] = p.normal_y;
			rendered_normals[4*rendered_vector_index+2] = p.normal_z;
			rendered_normals[4*rendered_vector_index+3] = 0;
		}
	}

	// weights based on rendered image
	cv::Mat rendered_weights_trivial(rendered_image_for_dense_color_.size(), CV_32FC1, cv::Scalar::all(1.0));

	opencl_optimize_->prepareRenderedAndErrorBuffers(
		render_proj_f_[0], render_proj_f_[1], render_proj_c_[0], render_proj_c_[1],
		render_rect_.x, render_rect_.y, render_rect_.width, render_rect_.height,
		rendered_points.data(), rendered_normals.data(), (float*) rendered_image_for_dense_color_.data, (float*) rendered_weights_trivial.data);
}

// new version, takes render buffers as input
template <typename Scalar>
void ICPCombinedFunctor<Scalar>::initOpenCLRender(const RenderBuffers& render_buffers)
{
	// weights based on rendered image
	//cv::Mat rendered_weights_trivial(render_rect_.size(), CV_32FC1, cv::Scalar::all(1.0));
	// instead weights as an image buffer
	std::vector<float> rendered_weights_trivial(render_rect_.width * render_rect_.height, 1.0);
	ImageBuffer rendered_weights_image_buffer(opencl_images_->getCL());
	rendered_weights_image_buffer.writeFromFloatVector(rendered_weights_trivial);

	if (params.combined_image_error != Parameters::IMAGE_ERROR_YCBCR) {
		throw new std::exception ("not implemented");
	}
	ImageBuffer rendered_image_ycrcb = opencl_images_->extractYCrCbFloat(render_buffers.getColorAsImageBuffer(), render_buffers.getWidth(), render_buffers.getHeight());

	opencl_optimize_->prepareRenderedAndErrorBuffersWithBuffers(
		render_proj_f_[0], render_proj_f_[1], render_proj_c_[0], render_proj_c_[1],
		render_rect_.x, render_rect_.y, render_rect_.width, render_rect_.height,
		render_buffers.getBufferRenderPoints(), render_buffers.getBufferRenderNormals(), rendered_image_ycrcb.getBuffer(), rendered_weights_image_buffer.getBuffer());
}

template <typename Scalar>
Eigen::Matrix4f ICPCombinedFunctor<Scalar>::xToMatrix4f(const InputType& x) const {
	Eigen::VectorXf params = x.cast<float> ();
	pcl::WarpPointRigid6D<PointT, PointT> warp_point;
	warp_point.setParam (params);
	return warp_point.getTransform();
}


template <typename Scalar>
Eigen::Affine3f ICPCombinedFunctor<Scalar>::xToAffine3f(const InputType& x) const {
	return Eigen::Affine3f(xToMatrix4f(x));
}

template <typename Scalar>
int ICPCombinedFunctor<Scalar>::errorPoints() const {
	return render_rect_.area();
}

template <typename Scalar>
int ICPCombinedFunctor<Scalar>::valuesICPAndColor() const {
	return (errorPoints() * (1 + getImageChannelCount()));
}

template <typename Scalar>
int ICPCombinedFunctor<Scalar>::values() const {
	int result = valuesICPAndColor();
	return result;
}

template <typename Scalar>
void ICPCombinedFunctor<Scalar>::combinedDebugImages(Eigen::Affine3f const& transform, std::vector<float> const& error_vector, bool new_ordering) const
{
	cv::Mat icp_error_image(render_rect_.height, render_rect_.width, CV_32FC1, cv::Scalar::all(0.5));
	// make general:
	std::vector<cv::Mat> image_error_vec;
	for (int i = 0; i < getImageChannelCount(); i++) {
		image_error_vec.push_back(cv::Mat(render_rect_.height, render_rect_.width, CV_32FC1, cv::Scalar::all(0.5)));
	}

	cv::Mat rendered_reference_image(render_rect_.height, render_rect_.width, CV_8UC3, cv::Scalar::all(0));
	for (int row = 0; row < icp_error_image.rows; row++) {
		for (int col = 0; col < icp_error_image.cols; col++) {
			if (new_ordering) {
				// new coallesced indexing
				int error_vector_index_icp = row * icp_error_image.cols + col;
				icp_error_image.at<float>(row, col) += error_vector[error_vector_index_icp];
				for (int i = 0; i < getImageChannelCount(); i++) {
					int error_vector_index_image = (i+1) * errorPoints() + row * icp_error_image.cols + col;
					image_error_vec[i].at<float>(row, col) += error_vector[error_vector_index_image];
				}
			}
			else {
				// old indexing scheme:
				int error_vector_index_icp = getErrorChannelCount() * (row * icp_error_image.cols + col);
				icp_error_image.at<float>(row, col) += error_vector[error_vector_index_icp];
				for (int i = 0; i < getImageChannelCount(); i++) {
					int error_vector_index_image = getErrorChannelCount() * (row * icp_error_image.cols + col) + 1 + i;
					image_error_vec[i].at<float>(row, col) += error_vector[error_vector_index_image];
				}
			}

			// also fill in ref image
			if (rendered_cloud_with_normals_ && rendered_cloud_with_normals_->size() == render_rect_.area()) {
				const PointICPTargetT& p = rendered_cloud_with_normals_->at(col, row);
				cv::Vec3b& ref_pixel = rendered_reference_image.at<cv::Vec3b>(row, col);
				ref_pixel[0] = p.b;
				ref_pixel[1] = p.g;
				ref_pixel[2] = p.r;
			}
			else {
				cv::Vec3b& ref_pixel = rendered_reference_image.at<cv::Vec3b>(row, col);
				ref_pixel[0] = 0;
				ref_pixel[1] = 0;
				ref_pixel[2] = 0;
			}
		}
	}


	cv::Mat frame_image_roi = frame_image_for_dense_color_vec_[initialized_octave_](frame_object_rect_vec_[initialized_octave_]);

	std::vector<cv::Mat> frame_image_channels;
	cv::split(frame_image_roi, frame_image_channels);

	std::vector<cv::Mat> frame_object_image_vec;
	for (int i = 0; i < getImageChannelCount(); i++) {
		frame_object_image_vec.push_back(cv::Mat());
		frame_image_channels[i].copyTo(frame_object_image_vec.back(), frame_object_mask_vec_[initialized_octave_](frame_object_rect_vec_[initialized_octave_]));
	}

	std::vector<cv::Mat> v_images;
	v_images.push_back(floatC1toCharC3(icp_error_image));
	for (int i = 0; i < getImageChannelCount(); i++) {
		v_images.push_back(floatC1toCharC3(image_error_vec[i]));
	}
	v_images.push_back(rendered_reference_image);
	for (int i = 0; i < getImageChannelCount(); i++) {
		// scale frame images to match render resolution here
		cv::Mat source_image = frame_object_image_vec[i];
		cv::Mat image_to_show = source_image;
		cv::Size desired_size = v_images[0].size();
		if (source_image.size() != desired_size) {
			cv::resize(source_image, image_to_show, desired_size, 0, 0, cv::INTER_NEAREST);
		}
		v_images.push_back(floatC1toCharC3(image_to_show));
	}
	cv::Mat combined_error_images = createMxN(2, 1 + getImageChannelCount(), v_images);
	float scale = 1 << initialized_octave_;
	scale *= params.combined_debug_images_scale;
	cv::Mat combined_error_images_larger;
	cv::resize(combined_error_images, combined_error_images_larger, cv::Size(), scale, scale, cv::INTER_NEAREST); 
	showInWindow("Combined Error Images (functor)", combined_error_images_larger);

	///////////////////
	// A second window of debug?
	// too lazy to generalize this right now
	if (initialized_octave_ == 0 && rendered_cloud_with_normals_ && rendered_cloud_with_normals_->size() == render_rect_.area()) {
		v_images.clear();

		cv::Mat frame_under_render = frame_.image_color(render_rect_).clone();

		v_images.push_back(frame_under_render);

		// project the point cloud by transform
		CloudICPTargetT::Ptr transformed_render_cloud(new CloudICPTargetT);
		pcl::transformPointCloud(*rendered_cloud_with_normals_, *transformed_render_cloud, transform);
		CloudICPTargetT::Ptr projected_cloud = projectRenderCloud(*transformed_render_cloud, render_proj_f_, render_proj_c_, Eigen::Vector2f(render_rect_.x, render_rect_.y));
		cv::Mat projected_image = cloudToImage(*projected_cloud);
		v_images.push_back(projected_image);

		cv::Mat both = frame_under_render * 0.5 + projected_image * 0.5;
		v_images.push_back(both);

		cv::Mat cdi_second = createMxN(1,3,v_images);
		showInWindow("CDI Second Window", cdi_second);

		// only save in here right now....again, lazy...and hacky
		if (params.save_cdi_images) {
			v_images.clear();
			//v_images.push_back(floatC1toCharC3(icp_error_image));
			for (int i = 0; i < getImageChannelCount(); i++) {
				v_images.push_back(floatC1toCharC3(image_error_vec[i]));
			}
			v_images.push_back(both);
			cv::Mat what_you_decided_to_save = createMxN(1,v_images.size(),v_images);
			combined_debug_image_v.push_back(what_you_decided_to_save);
		}
	}




	if (params.combined_pause_every_eval) cv::waitKey();
	else cv::waitKey(1);
}

template <typename Scalar>
template <typename Derived>
void ICPCombinedFunctor<Scalar>::errorICPAndColorOpenCL (const InputType &x, Eigen::MatrixBase<Derived> const &partial_fvec) const
{
	if (!initRenderCalled_) throw new exception ("!initRenderCalled_");
	if (initialized_octave_ < 0) throw new exception ("initialized_octave_ < 0");

	Eigen::Affine3f x_transform = xToAffine3f(x);
	std::vector<float> error_vector(opencl_optimize_->getErrorVectorSize(), 0); // todo: don't init
	if (params.combined_debug_single_kernel) {
		Eigen::Matrix<float,6,6> LHS;
		Eigen::Matrix<float,6,1> RHS;
		opencl_optimize_->computeErrorAndGradient(x_transform, LHS, RHS, error_vector.data(), NULL);
	}
	else {
		opencl_optimize_->errorICPAndColor(x_transform, error_vector.data());
	}
	// todo: this a better way
	// also I don't think this is right for combined_debug_single_kernel!!!
	for (size_t i = 0; i < values(); i++) {
		const_cast<Eigen::MatrixBase<Derived>&>(partial_fvec)[i] = error_vector[i];
	}

	// grab various statistics about the error
	// can be slow (~5ms)
	if (params.combined_compute_error_statistics) {
		pcl::ScopeTime st("[TIMING] compute_combined_errors_statistics in functor");
		float total_squared_icp_error = 0;
		float total_squared_color_error = 0;
		float max_squared_icp_error = 0;
		float max_squared_color_error = 0;
		int nonzero_icp_count = 0;
		int nonzero_color_count = 0;
		for (size_t i = 0; i < values() / getErrorChannelCount(); i++) {
			float this_icp_error = error_vector[getErrorChannelCount()*i];
			if (this_icp_error != 0) nonzero_icp_count++;
			float this_icp_error_squared = this_icp_error * this_icp_error;
			total_squared_icp_error += this_icp_error_squared;
			max_squared_icp_error = max(max_squared_icp_error, this_icp_error_squared);
			for (int c = 0; c < getImageChannelCount(); ++c) {
				float this_color_error = error_vector[getErrorChannelCount()*i + 1 + c];
				if (this_color_error != 0) nonzero_color_count++;
				float this_color_error_squared = this_color_error * this_color_error;
				total_squared_color_error += this_color_error_squared;
				max_squared_color_error = max(max_squared_color_error, this_color_error_squared);
			}
		}
		last_error_icp_ = sqrt(total_squared_icp_error);
		last_error_color_ = sqrt(total_squared_color_error);
		last_error_icp_max_ = sqrt(max_squared_icp_error);
		last_error_color_max_ = sqrt(max_squared_color_error);
		last_error_icp_count_ = nonzero_icp_count;
		last_error_color_count_ = nonzero_color_count;
	}

	//////////////////////////////
	// debug (will be slow)
	if (params.combined_debug_images) {
		combinedDebugImages(x_transform, error_vector, params.combined_debug_single_kernel);
	}
}

template <typename Scalar>
template <typename Derived>
void ICPCombinedFunctor<Scalar>::dfICPAndColorOpenCL(const InputType &x, Eigen::MatrixBase<Derived> const &partial_jmat) const 
{
	Eigen::Affine3f x_transform = xToAffine3f(x);
	std::vector<float> error_matrix(opencl_optimize_->getErrorMatrixSize(), 0); // todo: don't init
	if (params.combined_debug_single_kernel) {
		Eigen::Matrix<float,6,6> LHS;
		Eigen::Matrix<float,6,1> RHS;
		opencl_optimize_->computeErrorAndGradient(x_transform, LHS, RHS, NULL, error_matrix.data());

		// In the new version, it's by channel, col, row
		// and error is packed channel, row
		int error_channels = getErrorChannelCount();
		int error_point_count = errorPoints();
		for (int c = 0; c < error_channels; ++c) {
			for (int row = 0; row < error_point_count; ++row) {
				for (int col = 0; col < 6; ++col) {
					const_cast<Eigen::MatrixBase<Derived>&>(partial_jmat)(c*error_point_count+row, col) = 
						error_matrix[c*error_point_count*6 + col*error_point_count + row];
				}
			}
		}
	}
	else {
		opencl_optimize_->dfICPAndColor(x_transform, error_matrix.data());
		for (int row = 0; row < values(); row++) {
			for (int col = 0; col < 6; col++) {
				const_cast<Eigen::MatrixBase<Derived>&>(partial_jmat)(row, col) = error_matrix[row * 6 + col];
			}
		}
	}
}

// the main value function for EigenLM
template <typename Scalar>
int ICPCombinedFunctor<Scalar>::operator() (const InputType &x, ValueType &fvec) const
{
	pcl::StopWatch sw;

	fvec.setZero();
	size_t fvec_start_index = 0;

	//////////////////////////////
	// Combined ICP and Color
	last_error_icp_ = 0;
	last_error_color_ = 0;
	size_t icp_and_color_error_count = 0;
	icp_and_color_error_count = valuesICPAndColor();
	errorICPAndColorOpenCL(x, fvec.segment(fvec_start_index, icp_and_color_error_count));
	fvec_start_index += icp_and_color_error_count;

	last_error_total_ = sqrt((last_error_icp_*last_error_icp_) + (last_error_color_*last_error_color_));
	if (params.combined_verbose) {
		cout << "total_icp_error: " << last_error_icp_ << endl;
		cout << "total_dense_color_error: " << last_error_color_ << endl;
		cout << "total error: " << last_error_total_ << endl;
	}

	rs_f.push(sw.getTime());

	return 0;
}

template <typename Scalar>
int ICPCombinedFunctor<Scalar>::df (const InputType &x, JacobianType &jmat) const
{
	pcl::StopWatch sw;

	// initialize jacobian matrix
	jmat.setZero();
	int jmat_start_index = 0;

	////////////////////////////////
	// ICP and Color combined jacobian
	int icp_and_color_error_count = 0;
	icp_and_color_error_count = valuesICPAndColor();
	dfICPAndColorOpenCL(x, jmat.block(jmat_start_index, 0, icp_and_color_error_count, 6));
	jmat_start_index += icp_and_color_error_count;

	rs_df.push(sw.getTime());

	return 0;
}

// try a CPU implementation of Gauss Newton before messing with OpenCL
// This is quite old:
template <typename Scalar>
void ICPCombinedFunctor<Scalar>::solveGaussNewton(const InputType &x, InputType &x_result, int& iterations, ValueType& error_vector) {
	x_result = x;
	JacobianType J(values(), 6);
	error_vector = ValueType(values());

	for (iterations = 0; iterations < params.combined_gauss_newton_max_iterations; iterations++) {
		operator() (x_result, error_vector);
		df(x_result, J);
		Eigen::MatrixXf LHS = (J.transpose()*J).cast<float>();
		Eigen::VectorXf RHS = -(J.transpose()*error_vector).cast<float>();
		InputType x_delta = (LHS.colPivHouseholderQr().solve(RHS)).cast<Scalar>();
		x_result += x_delta;
	}
}

// Try GPU based gauss newton (reduction within computation, avoid global memory writes)
template <typename Scalar>
void ICPCombinedFunctor<Scalar>::solveGaussNewtonGPUFull(const InputType &x, InputType &x_result, int& iterations) {
	x_result = x;

	for (iterations = 0; iterations < params.combined_gauss_newton_max_iterations; iterations++) {
		Eigen::Matrix<float,6,6> LHS;
		Eigen::Matrix<float,6,1> RHS;
		Eigen::Affine3f x_transform = xToAffine3f(x_result);

		std::vector<float> error_vector;
		float* error_vector_ptr = NULL;
		bool retrieve_error_vector = params.combined_debug_images; // or other things
		if (retrieve_error_vector) {
			error_vector.resize(opencl_optimize_->getErrorVectorSize());
			error_vector_ptr = error_vector.data();
		}

		opencl_optimize_->computeErrorAndGradient(x_transform, LHS, RHS, error_vector_ptr, NULL);

		// assumes we retrieved error vector based on above logic
		if (retrieve_error_vector) {
			combinedDebugImages(x_transform, error_vector, true);
		}


		if (params.combined_verbose) {
			cout << "LHS:\n" << LHS << endl;
			cout << "RHS:\n" << RHS.transpose() << endl;
		}

		InputType x_delta = LHS.ldlt().solve(-RHS).cast<Scalar>();
		x_result += x_delta;

		if (params.combined_verbose) {
			cout << "x_delta:\n" << x_delta.transpose() << endl;
			cout << "x_result:\n" << x_result.transpose() << endl;
		}




		/////////// continue?
		if (params.combined_gauss_newton_min_delta_to_continue > 0) {
			// get the max(abs(x_delta))
			Scalar max_component = x_delta.array().abs().maxCoeff();
			if (max_component < params.combined_gauss_newton_min_delta_to_continue) break;
		}
	}
}
