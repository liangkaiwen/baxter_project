#include "model_k_means.h"

#include "EigenUtilities.h"

#include "opencv_utilities.h"

#include "MeshUtilities.h"

#include "util.h"

#include <boost/assign.hpp>


// perhaps we just instantiate kernels when needed here?
#include "KernelDivideFloats.h"
#include "KernelExtractVolumeSlice.h"
#include "KernelExtractVolumeSliceFloat4.h"
#include "KernelExtractVolumeSliceFloat4Length.h"
#include "KernelExtractVolumeFloat.h"
#include "KernelAddFrame.h"
#include "KernelAddFrameTo2Means.h"
#include "KernelAddFrameTo2MeansUsingNormals.h"
#include "KernelAddFrameTo2MeansUsingStoredNormals.h"
#include "KernelAddFrameIfCompatible.h"
#include "KernelRender2MeansAbs.h"
#include "KernelRenderPointsAndNormals.h"
#include "KernelDotVolumeNormal.h"
#include "KernelMinAbsVolume.h"
#include "KernelAddVolumes.h"
#include "KernelAddFloatsWithWeights.h"
#include "KernelAddFloatsWithWeightsExternalWeight.h"
#include "KernelMinAbsFloatsWithWeights.h"
#include "KernelMinAbsFloatsWithWeightsRecordIndex.h"
#include "KernelMinAbsFloatsWithWeightsAndMinimumWeightFraction.h"
#include "KernelBetterNormal.h"
#include "KernelNormalsToColorImage.h"
#include "KernelNormalsToShadedImage.h"
#include "KernelExtractFloat4ForPointImage.h"
#include "KernelExtractIntForPointImage.h"
#include "KernelApplyPoseToNormals.h"
#include "KernelComputeNormalVolume.h"
#include "KernelComputeNormalVolumeWithWeights.h"
#include "KernelComputeNormalVolumeWithWeightsUnnormalized.h"
#include "KernelMaxFloats.h"
#include "KernelPickIfIndexFloats.h"
#include "KernelPickIfIndexFloat4.h"
#include "KernelMarkPointsViolateEmpty.h"
#include "KernelSetUChar.h"




std::vector<Eigen::Vector3f> ModelKMeans::fixed_normal_list_ = boost::assign::list_of
	(Eigen::Vector3f(1,0,0))
	(Eigen::Vector3f(-1,0,0))
	(Eigen::Vector3f(0,1,0))
	(Eigen::Vector3f(0,-1,0))
	(Eigen::Vector3f(0,0,1))
	(Eigen::Vector3f(0,0,-1))
	.convert_to_container<std::vector<Eigen::Vector3f> > () ;

std::vector<Eigen::AngleAxisf> ModelKMeans::fixed_normal_cameras_ = boost::assign::list_of
	(Eigen::AngleAxisf(M_PI / 2, Eigen::Vector3f(0, -1, 0)))
	(Eigen::AngleAxisf(M_PI / 2, Eigen::Vector3f(0, 1, 0)))
	(Eigen::AngleAxisf(M_PI / 2, Eigen::Vector3f(1, 0, 0)))
	(Eigen::AngleAxisf(M_PI / 2, Eigen::Vector3f(-1, 0, 0)))
	(Eigen::AngleAxisf(M_PI, Eigen::Vector3f(0, -1, 0)))
	(Eigen::AngleAxisf(0, Eigen::Vector3f(0, -1, 0)))
	.convert_to_container<std::vector<Eigen::AngleAxisf> > () ;

std::vector<Eigen::Array4ub> ModelKMeans::fixed_color_list_ = boost::assign::list_of
	(Eigen::Array4ub(255,0,0, 255))
	(Eigen::Array4ub(0,255,0, 255))
	(Eigen::Array4ub(0,0,255, 255))
	(Eigen::Array4ub(255,255,0, 255))
	(Eigen::Array4ub(255,0,255, 255))
	(Eigen::Array4ub(0,255,255, 255))
	.convert_to_container<std::vector<Eigen::Array4ub> > () ;

ModelKMeans::ModelKMeans(boost::shared_ptr<OpenCLAllKernels> all_kernels, VolumeModelerAllParams const& params, boost::shared_ptr<Alignment> alignment_ptr, RenderBuffers const& render_buffers)
	: ModelBase(all_kernels, params, alignment_ptr, render_buffers),
	trackbar_window_(params.volume.cell_count[1])
{
	reset();
}

// not deep
ModelKMeans* ModelKMeans::clone()
{
	return new ModelKMeans(*this);
}

void ModelKMeans::reset()
{
	buffer_mean_list_.resize(params_.model_k_means.k);
	buffer_count_list_.resize(params_.model_k_means.k);
	if (params_.model_k_means.store_normals) buffer_normal_list_.resize(params_.model_k_means.k);
	for(size_t i = 0; i < params_.model_k_means.k; ++i) {
		buffer_mean_list_[i].reset(new VolumeBuffer(all_kernels_));
		buffer_count_list_[i].reset(new VolumeBuffer(all_kernels_));

		buffer_mean_list_[i]->resize(params_.volume.cell_count, sizeof(float));
		buffer_count_list_[i]->resize(params_.volume.cell_count, sizeof(float));

		buffer_mean_list_[i]->setFloat(-1);
		buffer_count_list_[i]->setFloat(0);

		if (params_.model_k_means.store_normals) {
			// also normals
			buffer_normal_list_[i].reset(new VolumeBuffer(all_kernels_));
			buffer_normal_list_[i]->resize(params_.volume.cell_count, sizeof(float) * 4);
			buffer_normal_list_[i]->setFloat4(Eigen::Array4f::Zero());
		}

	}

	volume_pose_ = Eigen::Affine3f::Identity();
}

void ModelKMeans::renderModel(
        const ParamsCamera & params_camera,
        const Eigen::Affine3f & model_pose,
        RenderBuffers & render_buffers)
{
    render_buffers.setSize(params_camera.size.x(), params_camera.size.y());
	render_buffers.resetAllBuffers();

	if (params_.model_k_means.debug_rendering && !params_.volume_modeler.command_line_interface) {

		Eigen::Affine3f volume_pose_in_world = model_pose * volume_pose_;
		const size_t cell_count = buffer_mean_list_[0]->getSizeInCells();

		KernelRenderPointsAndNormals _KernelRenderPointsAndNormals(*all_kernels_);
		KernelDotVolumeNormal _KernelDotVolumeNormal(*all_kernels_);
		KernelMaxFloats _KernelMaxFloats(*all_kernels_);
		KernelAddFloatsWithWeights _KernelAddFloatsWithWeights(*all_kernels_);
		KernelAddFloatsWithWeightsExternalWeight _KernelAddFloatsWithWeightsExternalWeight(*all_kernels_);

		// try normal rendering of each volume as debug
		cv::Mat color, depth;

		// render each separately
		if (params_.model_k_means.render_all_6){
			boost::timer t;

			std::vector<cv::Mat> each_model_rendered;
			for (size_t i = 0; i < params_.model_k_means.k; ++i) {
                const int mask_value = 1;
				_KernelRenderPointsAndNormals.runKernel(
					*buffer_mean_list_[i],
					*buffer_count_list_[i],
					render_buffers.getImageBufferMask(),
					render_buffers.getImageBufferPoints(),
					render_buffers.getImageBufferNormals(),
					params_.volume.cell_size,
					volume_pose_in_world,
					params_camera.focal,
					params_camera.center,
					params_camera.min_max_depth[0],
					params_camera.min_max_depth[1],
					mask_value);
				render_buffers.getRenderPretty(color, depth);
				if (params_.model_k_means.k != 6) {
					std::string window_name = (boost::format("k_means_buffer_%d") % i).str();
					cv::imshow(window_name, depth);
				}
				else {
					Eigen::Array4ub & color = fixed_color_list_[i];
					cv::Mat depth_with_colored_border = getImageWithBorder(depth, 10, cv::Scalar(color[0], color[1], color[2]));
					each_model_rendered.push_back(depth_with_colored_border.clone());
				}
				render_buffers.resetAllBuffers();
			}
			if (each_model_rendered.size() == 6) {
				cv::Mat all_6 = createMxN(3,2,each_model_rendered);
				cv::Mat all_6_scaled;
				const float scale = params_.model_k_means.render_all_6_scale;
				cv::resize(all_6, all_6_scaled, cv::Size(), scale, scale);
				cv::imshow("all_6_scaled", all_6_scaled);
			}

			cout << "TIME render_all_6: " << t.elapsed() << endl;
		}

		if (params_.model_k_means.render_all_6_from_canonical && params_.model_k_means.k == 6) {
			boost::timer t;

			std::vector<cv::Mat> each_model_rendered;

			// when you did them at the end...maybe useful at some point...
			//UpdateInterface::PoseListPtrT camera_pose_list(new UpdateInterface::PoseListT);

			for (size_t i = 0; i < params_.model_k_means.k; ++i) {
				Eigen::Affine3f canonical_camera;
				canonical_camera = fixed_normal_cameras_[i];
				canonical_camera.translate(Eigen::Vector3f(0,0,-params_.model_k_means.render_all_6_from_canonical_distance));

				// get point at center of model (including volume pose)
				Eigen::Vector3f center_of_model_in_world = volume_pose_ * (Eigen::Vector3f(0.5, 0.5, 0.5).array() * params_.volume.cell_count.cast<float>() * params_.volume.cell_size).matrix();
				canonical_camera.pretranslate(center_of_model_in_world);

				// debug at end?
				//camera_pose_list->push_back(canonical_camera);

				if (update_interface_) {
					std::string name = (boost::format("canonical_camera_%d") % i).str();
					UpdateInterface::PoseListPtrT this_camera(new UpdateInterface::PoseListT);
					this_camera->push_back(canonical_camera);
					update_interface_->updateCameraList(name, this_camera);
					update_interface_->updateScale(name, 5);
					update_interface_->updateColor(name, fixed_color_list_[i]);
				}				

                const int mask_value = 1;
				_KernelRenderPointsAndNormals.runKernel(
					*buffer_mean_list_[i],
					*buffer_count_list_[i],
					render_buffers.getImageBufferMask(),
					render_buffers.getImageBufferPoints(),
					render_buffers.getImageBufferNormals(),
					params_.volume.cell_size,
					canonical_camera.inverse() * volume_pose_, // still not quite sure.  But this works, so whatever...
					params_camera.focal,
					params_camera.center,
					params_camera.min_max_depth[0],
					params_camera.min_max_depth[1],
					mask_value);
				render_buffers.getRenderPretty(color, depth);
				Eigen::Array4ub & color = fixed_color_list_[i];
				cv::Mat depth_with_colored_border = getImageWithBorder(depth, 10, cv::Scalar(color[0], color[1], color[2]));
				each_model_rendered.push_back(depth_with_colored_border.clone());
				render_buffers.resetAllBuffers();
			}
			if (each_model_rendered.size() == 6) {
				cv::Mat all_6 = createMxN(3,2,each_model_rendered);
				cv::Mat all_6_scaled;
				const float scale = params_.model_k_means.render_all_6_scale;
				cv::resize(all_6, all_6_scaled, cv::Size(), scale, scale);
				cv::imshow("render_all_6_from_canonical", all_6_scaled);
			}


			cout << "TIME render_all_6_from_canonical: " << t.elapsed() << endl;
		}


		// for making a combined volume, and rendering that...
		// note that this will fail once you want to add color (unless you track alongside)
		VolumeBuffer combined_d(all_kernels_, params_.volume.cell_count, sizeof(float));
		VolumeBuffer combined_dw(all_kernels_, params_.volume.cell_count, sizeof(float));

		if (params_.model_k_means.k == 6) {
			Eigen::Vector4f to_camera_in_camera(0,0,-1, 0);
			Eigen::Vector4f to_camera_in_world = volume_pose_in_world.inverse() * to_camera_in_camera;

			///////////
			// just select best entire volume at once
			{
				debug_best_last_frame_.assign(params_.model_k_means.k, false);

				int best_i = -1;
				float max_dot = -1;
				for (int i = 0; i < fixed_normal_list_.size(); ++i) {
					float this_dot = fixed_normal_list_[i].dot(to_camera_in_world.head<3>());
					if (best_i < 0 || this_dot > max_dot) {
						best_i = i;
						max_dot = this_dot;
					}
				}

				debug_best_last_frame_[best_i] = true;

				// render this one only?
                const int mask_value = 1;
				_KernelRenderPointsAndNormals.runKernel(
					*buffer_mean_list_[best_i],
					*buffer_count_list_[best_i],
					render_buffers.getImageBufferMask(),
					render_buffers.getImageBufferPoints(),
					render_buffers.getImageBufferNormals(),
					params_.volume.cell_size,
					volume_pose_in_world,
					params_camera.focal,
					params_camera.center,
					params_camera.min_max_depth[0],
					params_camera.min_max_depth[1],
					mask_value);
				render_buffers.getRenderPretty(color, depth);
				std::string window_name = (boost::format("most compatible only")).str();
				cv::Mat depth_with_border = getImageWithBorder(depth, 5, eigenToCVColor(fixed_color_list_[best_i]));
				cv::imshow(window_name, depth_with_border);
				render_buffers.resetAllBuffers();
			}


			///////////
			// or add up all those volumes with sufficiently good dot product with current?
			{
				combined_d.setFloat(-1);
				combined_dw.setFloat(0);
				float min_dot_product = cos(params_.model_k_means.compatibility_render_max_angle_degrees * M_PI / 180);
				debug_compatible_last_frame_.assign(6, false);
				for (int i = 0; i < fixed_normal_list_.size(); ++i) {
					float this_dot = fixed_normal_list_[i].dot(to_camera_in_world.head<3>());
					if (this_dot >= min_dot_product) {
						debug_compatible_last_frame_[i] = true;

						_KernelAddFloatsWithWeights.runKernel(
							combined_d.getBuffer(), combined_dw.getBuffer(),
							buffer_mean_list_[i]->getBuffer(), buffer_count_list_[i]->getBuffer(),
							cell_count);

#if 0
						// debug render as building
						_KernelRenderPointsAndNormals.runKernel(
							combined_d,
							combined_dw,
							render_buffers.getImageBufferMask(),
							render_buffers.getImageBufferPoints(),
							render_buffers.getImageBufferNormals(),
							params_.volume.cell_size,
							volume_pose_in_world,
							params_camera.focal,
							params_camera.center,
							params_camera.min_max_depth[0],
							params_camera.min_max_depth[1],
							mask_value);
						render_buffers.getRenderPretty(color, depth);
						std::string window_name = (boost::format("debug_compatible_%d") % i).str();
						cv::imshow(window_name, depth);
						render_buffers.resetAllBuffers();
#endif
					}
				}
                const int mask_value = 1;
				_KernelRenderPointsAndNormals.runKernel(
					combined_d,
					combined_dw,
					render_buffers.getImageBufferMask(),
					render_buffers.getImageBufferPoints(),
					render_buffers.getImageBufferNormals(),
					params_.volume.cell_size,
					volume_pose_in_world,
					params_camera.focal,
					params_camera.center,
					params_camera.min_max_depth[0],
					params_camera.min_max_depth[1],
					mask_value);
				render_buffers.getRenderPretty(color, depth);
				std::string window_name = (boost::format("sum of compatible")).str();
				// border with all used...
				cv::Mat previous_depth_border = depth;
				for (size_t i = 0; i < fixed_normal_list_.size(); ++i) {
					if (debug_compatible_last_frame_[i]) {
						cv::Mat new_depth_border = getImageWithBorder(previous_depth_border, 5, eigenToCVColor(fixed_color_list_[i]));
						previous_depth_border = new_depth_border;
					}
				}
				cv::imshow(window_name, previous_depth_border);
				render_buffers.resetAllBuffers();
			}

			///////////
			// include cos weighting
			{
				combined_d.setFloat(-1);
				combined_dw.setFloat(0);
				float min_dot_product = cos(params_.model_k_means.compatibility_render_max_angle_degrees * M_PI / 180);
				debug_compatible_last_frame_.assign(6, false);
				for (int i = 0; i < fixed_normal_list_.size(); ++i) {
					float this_dot = fixed_normal_list_[i].dot(to_camera_in_world.head<3>());
					if (this_dot >= min_dot_product) {
						debug_compatible_last_frame_[i] = true;

						float external_weight = this_dot;

						_KernelAddFloatsWithWeightsExternalWeight.runKernel(
							combined_d.getBuffer(), combined_dw.getBuffer(),
							buffer_mean_list_[i]->getBuffer(), buffer_count_list_[i]->getBuffer(),
							external_weight,
							cell_count);

#if 0
						// debug render as building
						_KernelRenderPointsAndNormals.runKernel(
							combined_d,
							combined_dw,
							render_buffers.getImageBufferMask(),
							render_buffers.getImageBufferPoints(),
							render_buffers.getImageBufferNormals(),
							params_.volume.cell_size,
							volume_pose_in_world,
							params_camera.focal,
							params_camera.center,
							params_camera.min_max_depth[0],
							params_camera.min_max_depth[1],
							mask_value);
						render_buffers.getRenderPretty(color, depth);
						std::string window_name = (boost::format("debug_compatible_%d") % i).str();
						cv::imshow(window_name, depth);
						render_buffers.resetAllBuffers();
#endif
					}
				}
                const int mask_value = 1;
				_KernelRenderPointsAndNormals.runKernel(
					combined_d,
					combined_dw,
					render_buffers.getImageBufferMask(),
					render_buffers.getImageBufferPoints(),
					render_buffers.getImageBufferNormals(),
					params_.volume.cell_size,
					volume_pose_in_world,
					params_camera.focal,
					params_camera.center,
					params_camera.min_max_depth[0],
					params_camera.min_max_depth[1],
					mask_value);
				render_buffers.getRenderPretty(color, depth);
				std::string window_name = (boost::format("cos weighted sum of compatible")).str();
				// border with all used...
				cv::Mat previous_depth_border = depth;
				for (size_t i = 0; i < fixed_normal_list_.size(); ++i) {
					if (debug_compatible_last_frame_[i]) {
						cv::Mat new_depth_border = getImageWithBorder(previous_depth_border, 5, eigenToCVColor(fixed_color_list_[i]));
						previous_depth_border = new_depth_border;
					}
				}
				cv::imshow(window_name, previous_depth_border);
				render_buffers.resetAllBuffers();
			}



#if 0
			/////////////////////////
			// try to minimize down to 2 values per voxel
			{
				VolumeBuffer best_dot_product_value(*all_kernels_);
				best_dot_product_value.resize(params_.volume.cell_count, sizeof(float));
				best_dot_product_value.setFloat(-1);
				VolumeBuffer best_dot_product_index(*all_kernels_);
				best_dot_product_index.resize(params_.volume.cell_count, sizeof(int));
				best_dot_product_index.setInt(-1);

				for (int i = 0; i < fixed_normal_list_.size(); ++i) {
					VolumeBuffer dot_product_with_fixed_normal (*all_kernels_);
					dot_product_with_fixed_normal.resize(params_.volume.cell_count, sizeof(float));
					_KernelDotVolumeNormal.runKernel(*buffer_mean_list_[i], dot_product_with_fixed_normal, fixed_normal_list_[i]);
					_KernelMaxFloats.runKernel(best_dot_product_value.getBuffer(), best_dot_product_index.getBuffer(), dot_product_with_fixed_normal.getBuffer(), i, cell_count);
				}

				// now, uh...basically want the "compatible" volume and the "incompatible" volume with respect to this stupid max-agreement volume
				// so do the "picker" and pick out the local normals from the most compatible volume?

				// so it's stupid to compute normals once for the dot product, and then again to actually pick them

				//KernelPickIfIndexFloats _KernelPickIfIndexFloats(*all_kernels_);
				KernelPickIfIndexFloat4 _KernelPickIfIndexFloat4(*all_kernels_);
				// so now, wastefully, compute normal volume for each, and then pick the normal if compatible the max-compatible normal
				// sigh
				KernelComputeNormalVolume _KernelComputeNormalVolume(*all_kernels_);

				VolumeBuffer best_dot_product_normal(*all_kernels_);
				best_dot_product_normal.resize(params_.volume.cell_count, sizeof(float) * 4);
				best_dot_product_normal.setFloat4(Eigen::Array4f(0,0,0,0));

				for (int i = 0; i < fixed_normal_list_.size(); ++i) {
					VolumeBuffer normals_for_this_volume (*all_kernels_);
					_KernelComputeNormalVolume.runKernel(*buffer_mean_list_[i], normals_for_this_volume);
					// I think maybe this is right????
					_KernelPickIfIndexFloat4.runKernel(best_dot_product_normal.getBuffer(), best_dot_product_index.getBuffer(), normals_for_this_volume.getBuffer(), i, cell_count);
				}


				// and now, at long last combine volumes which are compatible with this normal
				// somehow...somehow...similar to add if compatible
				// this is so wrong and stupid
				// totally not going to work.
				// going to arbitrarily split points between volumes until they're not meaningful

				VolumeBuffer combined_compatible_mean(*all_kernels_);
				combined_compatible_mean.resize(params_.volume.cell_count, sizeof(float));
				combined_compatible_mean.setFloat(-1);
				VolumeBuffer combined_compatible_count(*all_kernels_);
				combined_compatible_count.resize(params_.volume.cell_count, sizeof(float));
				combined_compatible_count.setFloat(0);

				for (int i = 0; i < fixed_normal_list_.size(); ++i) {
					// need add floats if compatible locally with other stupid volume
					// todo etc

				}
			}
#endif




#if 0
			/////////////
			// another idea...select which has local normal most in agreement with its supposed fixed normal
			// so this is clearly wrong on its own, but it does tend to pick reliable values at least
			// of course, weight is perhaps a better measure of reliability
			{
				VolumeBuffer best_dot_product_value(*all_kernels_);
				best_dot_product_value.resize(params_.volume.cell_count, sizeof(float));
				best_dot_product_value.setFloat(-1);
				VolumeBuffer best_dot_product_index(*all_kernels_);
				best_dot_product_index.resize(params_.volume.cell_count, sizeof(int));
				best_dot_product_index.setInt(-1);

				for (int i = 0; i < fixed_normal_list_.size(); ++i) {
					VolumeBuffer dot_product_with_fixed_normal (*all_kernels_);
					dot_product_with_fixed_normal.resize(params_.volume.cell_count, sizeof(float));
					_KernelDotVolumeNormal.runKernel(*buffer_mean_list_[i], dot_product_with_fixed_normal, fixed_normal_list_[i]);
					_KernelMaxFloats.runKernel(best_dot_product_value.getBuffer(), best_dot_product_index.getBuffer(), dot_product_with_fixed_normal.getBuffer(), i, cell_count);
				}

				// now select values and weights...
				KernelPickIfIndexFloats kernel_picker(*all_kernels_);
				for (int i = 0; i < fixed_normal_list_.size(); ++i) {
					kernel_picker.runKernel(combined_d.getBuffer(), best_dot_product_index.getBuffer(), buffer_mean_list_[i]->getBuffer(), i, cell_count);
					kernel_picker.runKernel(combined_dw.getBuffer(), best_dot_product_index.getBuffer(), buffer_count_list_[i]->getBuffer(), i, cell_count);
				}

				// render
				_KernelRenderPointsAndNormals.runKernel(combined_d, combined_dw, render_buffers.getImageBufferMask(),
					render_buffers.getImageBufferPoints(),
					render_buffers.getImageBufferNormals(),
					params_.volume.cell_size,
					volume_pose_in_world,
					params_camera.focal,
					params_camera.center,
					params_camera.min_max_depth[0],
					params_camera.min_max_depth[1],
					mask_value);
				render_buffers.getRenderPretty(color, depth);
				cv::imshow("local normal most in agreement with its supposed fixed normal", depth);
				render_buffers.resetAllBuffers();
			}
#endif


			/////////// slices!
			if (params_.model_k_means.debug_slices){
				boost::timer t;

				KernelExtractVolumeSlice _KernelExtractVolumeSlice(*all_kernels_);
				// also for stupid normals:
				KernelExtractVolumeSliceFloat4 _KernelExtractVolumeSliceFloat4(*all_kernels_);
				KernelExtractVolumeSliceFloat4Length _KernelExtractVolumeSliceFloat4Length(*all_kernels_);
				KernelNormalsToColorImage _KernelNormalsToColorImage(*all_kernels_);

				int axis = params_.model_k_means.slice_axis;
				int position = trackbar_window_.getTrackbarValue();

				// "always" get core values?
				{
					std::vector<cv::Mat> slices_colored;

					// usual mean volume slices
					{
						const float range = std::max(params_.volume.min_truncation_distance, params_.model_k_means.slice_color_max);

						for (int i = 0; i < buffer_mean_list_.size(); ++i) {
							ImageBuffer slice(all_kernels_->getCL());
							_KernelExtractVolumeSlice.runKernel(*buffer_mean_list_[i], axis, position, slice);
							//slices.push_back(slice.getMat());
							cv::Mat slice_mat = slice.getMat();
							cv::Mat slice_colored(slice_mat.size(), CV_8UC3);
							cv::MatConstIterator_<float> iter_in = slice_mat.begin<float>();
							cv::MatIterator_<cv::Vec3b> iter_out =slice_colored.begin<cv::Vec3b>();
							for (; iter_in != slice_mat.end<float>(); ++iter_in, ++iter_out) {
								float v = *iter_in;
								cv::Vec3b color;
								if (v >= 0) {
									int color_value = std::min(255 * v / range, 255.f);
									color = cv::Vec3b(0, color_value, 0);
								}
								else {
									int color_value = std::min(255 * -v / range, 255.f);
									color = cv::Vec3b(0, 0, color_value);
								}
								*iter_out = color;
							}
							slices_colored.push_back(getImageWithBorder(slice_colored, 10, eigenToCVColor(fixed_color_list_[i])));
						}
					}
					{
						cv::Mat all_6 = createMxN(3,2,slices_colored);
						cv::Mat all_6_scaled;
						//  cv::imshow("slices", all_6);
						cv::resize(all_6, all_6_scaled, cv::Size(), params_.model_k_means.slice_images_scale, params_.model_k_means.slice_images_scale);
						//cv::imshow("all_6_scaled", all_6_scaled);

						// also try trackbar window
						trackbar_window_.setMat(all_6_scaled);
					}
				}

				// normals
				{
					std::vector<cv::Mat> slices_colored;

					// this is very stupid...do something else when you have time
					{
						//KernelComputeNormalVolume _KernelComputeNormalVolume(*all_kernels_);
						KernelComputeNormalVolumeWithWeights _KernelComputeNormalVolumeWithWeights(*all_kernels_);


						for (int i = 0; i < buffer_mean_list_.size(); ++i) {
							VolumeBuffer this_volume_normals(all_kernels_);
							_KernelComputeNormalVolumeWithWeights.runKernel(*buffer_mean_list_[i], *buffer_count_list_[i], this_volume_normals);
							ImageBuffer this_volume_normals_slice(all_kernels_->getCL());
							_KernelExtractVolumeSliceFloat4.runKernel(this_volume_normals, axis, position, this_volume_normals_slice);
							ImageBuffer this_volume_normals_colored(all_kernels_->getCL());
							_KernelNormalsToColorImage.runKernel(this_volume_normals_slice, this_volume_normals_colored);
							cv::Mat this_slice_colored = this_volume_normals_colored.getMat();
							slices_colored.push_back(getImageWithBorder(this_slice_colored, 10, eigenToCVColor(fixed_color_list_[i])));
						}

					}
					{
						cv::Mat all_6 = createMxN(3,2,slices_colored);
						cv::Mat all_6_scaled;
						//  cv::imshow("slices", all_6);
						cv::resize(all_6, all_6_scaled, cv::Size(), params_.model_k_means.slice_images_scale, params_.model_k_means.slice_images_scale);
						cv::imshow("slices_normals", all_6_scaled);
					}
				}

				// normal magnitude (debug definitely)
				{
					std::vector<cv::Mat> slices_colored;

					// this is very stupid...do something else when you have time
					{
						//KernelComputeNormalVolume _KernelComputeNormalVolume(*all_kernels_);
						//KernelComputeNormalVolumeWithWeights _KernelComputeNormalVolumeWithWeights(*all_kernels_);
						KernelComputeNormalVolumeWithWeightsUnnormalized _KernelComputeNormalVolumeWithWeightsUnnormalized(*all_kernels_);


						for (int i = 0; i < buffer_mean_list_.size(); ++i) {
							VolumeBuffer this_volume_normals(all_kernels_);
							_KernelComputeNormalVolumeWithWeightsUnnormalized.runKernel(*buffer_mean_list_[i], *buffer_count_list_[i], this_volume_normals);
							ImageBuffer this_volume_normals_length_slice(all_kernels_->getCL());
							_KernelExtractVolumeSliceFloat4Length.runKernel(this_volume_normals, axis, position, this_volume_normals_length_slice);

							// gotta color somehow...
							cv::Mat this_volume_normals_length_slice_mat = this_volume_normals_length_slice.getMat();
							double min, max;
							cv::minMaxLoc(this_volume_normals_length_slice_mat, &min, &max);

							cout << "Debug remove max normal length: " << max << endl;

							cv::Mat max_to_1;
							this_volume_normals_length_slice_mat.convertTo(max_to_1, CV_8U, 255/max);
							cv::cvtColor(max_to_1, max_to_1, CV_GRAY2BGR);

							slices_colored.push_back(getImageWithBorder(max_to_1, 10, eigenToCVColor(fixed_color_list_[i])));
						}

					}
					{
						cv::Mat all_6 = createMxN(3,2,slices_colored);
						cv::Mat all_6_scaled;
						//  cv::imshow("slices", all_6);
						cv::resize(all_6, all_6_scaled, cv::Size(), params_.model_k_means.slice_images_scale, params_.model_k_means.slice_images_scale);
						cv::imshow("slices_normals_length", all_6_scaled);
					}
				}


				// weights
				{
					std::vector<cv::Mat> slices_colored;
					for (int i = 0; i < buffer_mean_list_.size(); ++i) {
						ImageBuffer slice(all_kernels_->getCL());
						_KernelExtractVolumeSlice.runKernel(*buffer_count_list_[i], axis, position, slice);
						cv::Mat slice_mat = slice.getMat();
						cv::Mat slice_colored(slice_mat.size(), CV_8UC3);
						cv::MatConstIterator_<float> iter_in = slice_mat.begin<float>();
						cv::MatIterator_<cv::Vec3b> iter_out =slice_colored.begin<cv::Vec3b>();
						for (; iter_in != slice_mat.end<float>(); ++iter_in, ++iter_out) {
							float v = *iter_in;
							cv::Vec3b color;
							if (v > 0) {
								color = cv::Vec3b(255,255,255);
							}
							else {
								color = cv::Vec3b(0, 0, 0);
							}
							*iter_out = color;
						}
						slices_colored.push_back(getImageWithBorder(slice_colored, 10, eigenToCVColor(fixed_color_list_[i])));
					}
					{
						cv::Mat all_6 = createMxN(3,2,slices_colored);
						cv::Mat all_6_scaled;
						cv::resize(all_6, all_6_scaled, cv::Size(), params_.model_k_means.slice_images_scale, params_.model_k_means.slice_images_scale);
						cv::imshow("slices_weights", all_6_scaled);
					}
				}



				cout << "TIME slices: " << t.elapsed() << endl;
			} // if (params_.model_k_means.debug_slices){


			KernelMarkPointsViolateEmpty _KernelMarkPointsViolateEmpty(*all_kernels_);
			KernelSetUChar _KernelSetUChar(*all_kernels_);

			if (params_.model_k_means.debug_meshes) {
				if (update_interface_) {
					std::vector<MeshPtr> individual_meshes = getPartialMeshes();

                    // Need this to show them, fool
                    //std::vector<MeshPtr> individual_meshes_normal = getValidPartialMeshesByNormal(individual_meshes);
                    //std::vector<MeshPtr> individual_meshes_violate = getValidPartialMeshesByEmptyViolation(individual_meshes);

#if 0
					for (size_t i = 0; i < buffer_mean_list_.size(); ++i) {
						std::string name = (boost::format("mesh_%d") % i).str();
						update_interface_->updateMesh(name, individual_meshes[i]);
					}
#endif

#if 0
					for (size_t i = 0; i < buffer_mean_list_.size(); ++i) {
						std::string name = (boost::format("mesh_normals_%d") % i).str();
						update_interface_->updateMesh(name, individual_meshes_normal[i]);
					}
#endif

#if 0
					for (size_t i = 0; i < buffer_mean_list_.size(); ++i) {
						std::string name = (boost::format("mesh_violate_%d") % i).str();
						update_interface_->updateMesh(name, individual_meshes_violate[i]);
					}
#endif

                    // boundary edges
#if 1
                    for (size_t i = 0; i < individual_meshes.size(); ++i) {
                        const Mesh & mesh = *individual_meshes[i];
                        std::string name = (boost::format("boundary_edges_%d") % i).str();
                        typedef std::vector<std::pair<int,int> > EdgeList;
                        EdgeList boundary_edges;
                        MeshUtilities::getBoundaryEdges(mesh, boundary_edges);
                        // flatten edges to actual vertices
                        MeshVertexVectorPtr boundary_lines_ptr(new MeshVertexVector);
                        BOOST_FOREACH(EdgeList::value_type & v, boundary_edges) {
                            boundary_lines_ptr->push_back(mesh.vertices[v.first]);
                            boundary_lines_ptr->push_back(mesh.vertices[v.second]);
                        }
                        update_interface_->updateLines(name, boundary_lines_ptr);
                    }
#endif



				}  // if update interface
			} // if (params_.model_k_means.debug_meshes) {

		} // if 6 volumes


		// simple average (KinectFusion style)
		{
			combined_d.setFloat(-1);
			combined_dw.setFloat(0);

			for (size_t i = 0; i < params_.model_k_means.k; ++i) {
				_KernelAddFloatsWithWeights.runKernel(combined_d.getBuffer(), combined_dw.getBuffer(), buffer_mean_list_[i]->getBuffer(), buffer_count_list_[i]->getBuffer(), cell_count);
			}
            const int mask_value = 1;
			_KernelRenderPointsAndNormals.runKernel(
				combined_d,
				combined_dw,
				render_buffers.getImageBufferMask(),
				render_buffers.getImageBufferPoints(),
				render_buffers.getImageBufferNormals(),
				params_.volume.cell_size,
				volume_pose_in_world,
				params_camera.focal,
				params_camera.center,
				params_camera.min_max_depth[0],
				params_camera.min_max_depth[1],
				mask_value);
			render_buffers.getRenderPretty(color, depth);
			cv::imshow("all volumes simple add", depth);
			render_buffers.resetAllBuffers();
		}


		// also do general min_abs kernel
		{
			boost::timer t;

			//KernelMinAbsFloatsWithWeights _KernelMinAbsFloatsWithWeights(*all_kernels_);
			KernelMinAbsFloatsWithWeightsRecordIndex _KernelMinAbsFloatsWithWeightsRecordIndex(*all_kernels_);

			combined_d.setFloat(-1);
			combined_dw.setFloat(0);

			// record index for debugging
			VolumeBuffer record_index(all_kernels_, params_.volume.cell_count, sizeof(int));
			record_index.setInt(-1); // not necessary if it works

			for (size_t i = 0; i < params_.model_k_means.k; ++i) {
				_KernelMinAbsFloatsWithWeightsRecordIndex.runKernel(
					combined_d.getBuffer(), combined_dw.getBuffer(), record_index.getBuffer(),
					buffer_mean_list_[i]->getBuffer(), buffer_count_list_[i]->getBuffer(), (int) i,
					combined_d.getSizeInBytes() / sizeof(float));
			}
            const int mask_value = 1;
			_KernelRenderPointsAndNormals.runKernel(
				combined_d,
				combined_dw,
				render_buffers.getImageBufferMask(),
				render_buffers.getImageBufferPoints(),
				render_buffers.getImageBufferNormals(),
				params_.volume.cell_size,
				volume_pose_in_world,
				params_camera.focal,
				params_camera.center,
				params_camera.min_max_depth[0],
				params_camera.min_max_depth[1],
				mask_value);
			render_buffers.getRenderPretty(color, depth);
			cv::imshow("KernelMinAbsFloatsWithWeightsRecordIndex", depth);

			{
				// before we reset the render buffers
				// also want to see the selected index
				ImageBuffer which_index(all_kernels_->getCL());
				KernelExtractIntForPointImage _KernelExtractIntForPointImage(*all_kernels_);
				_KernelExtractIntForPointImage.runKernel(record_index, render_buffers.getImageBufferPoints(), which_index, volume_pose_in_world, params_.volume.cell_size);
				cv::Mat which_index_mat = which_index.getMat();
				cv::Mat which_index_colored(which_index_mat.size(), CV_8UC3);
				cv::MatConstIterator_<int> iter_in;
				cv::MatIterator_<cv::Vec3b> iter_out;
				for (iter_in = which_index_mat.begin<int>(), iter_out = which_index_colored.begin<cv::Vec3b>(); 
					iter_in != which_index_mat.end<int>(); 
					++iter_in, ++iter_out) {
						int the_int = *iter_in;
						if (the_int >= 0 && the_int < fixed_color_list_.size()) {
							Eigen::Array4ub & color_eigen = fixed_color_list_[the_int];
							*iter_out = cv::Vec3b(color_eigen[0], color_eigen[1], color_eigen[2]);
						}
						else {
							*iter_out = cv::Vec3b(0,0,0);
						}
				}
				cv::imshow("which_index_colored KernelMinAbsFloatsWithWeightsRecordIndex", which_index_colored);
			}

			// now that we've used render_buffers.getImageBufferPoints(), can reset
			render_buffers.resetAllBuffers();

			cout << "TIME KernelMinAbsFloatsWithWeightsRecordIndex: " << t.elapsed() << endl;
		}

		// compare this with min_abs kernel including min_fraction of maximum weight
		// could redo this function to give us a debug view on that as well...
		{
			computeMinAbsVolume(params_.model_k_means.minimum_relative_count, combined_d, combined_dw);


			//// MAKE THIS A FUNCTION
			// and render
            const int mask_value = 1;
			_KernelRenderPointsAndNormals.runKernel(
				combined_d,
				combined_dw,
				render_buffers.getImageBufferMask(),
				render_buffers.getImageBufferPoints(),
				render_buffers.getImageBufferNormals(),
				params_.volume.cell_size,
				volume_pose_in_world,
				params_camera.focal,
				params_camera.center,
				params_camera.min_max_depth[0],
				params_camera.min_max_depth[1],
				mask_value);
			render_buffers.getRenderPretty(color, depth);
			cv::imshow("KernelMinAbsFloatsWithWeightsAndMinimumWeightFraction", depth);
			render_buffers.resetAllBuffers();
		}


		if (params_.model_k_means.store_normals) {
			KernelBetterNormal _KernelBetterNormal(*all_kernels_);

			// also try the "better normal" selector
			_KernelBetterNormal.runKernel(
				*buffer_mean_list_[0],
				*buffer_count_list_[0],
				*buffer_normal_list_[0],
				*buffer_mean_list_[1],
				*buffer_count_list_[1],
				*buffer_normal_list_[1],
				combined_d,
				combined_dw,
				volume_pose_in_world,
				params_.model_k_means.minimum_relative_count);

            const int mask_value = 1;
			_KernelRenderPointsAndNormals.runKernel(combined_d, combined_dw, render_buffers.getImageBufferMask(),
				render_buffers.getImageBufferPoints(),
				render_buffers.getImageBufferNormals(),
				params_.volume.cell_size,
				volume_pose_in_world,
				params_camera.focal,
				params_camera.center,
				params_camera.min_max_depth[0],
				params_camera.min_max_depth[1],
				mask_value);
			render_buffers.getRenderPretty(color, depth);
			cv::imshow("store_normals better normal", depth);
			render_buffers.resetAllBuffers();
		}

	} // debug rendering


	// the last one is the "official" render

	// replace reset with something actual...
	render_buffers.resetAllBuffers();


}

void ModelKMeans::updateModel(
	Frame & frame,
	const Eigen::Affine3f & model_pose)
{
	// do this before base class because won't be empty after that
	if (isEmpty()) {
		volume_pose_ = getSingleVolumePoseForFirstFrame(frame);
	}

	ModelBase::updateModel(frame, model_pose);

	ImageBuffer buffer_segments(all_kernels_->getCL());

	Eigen::Affine3f volume_pose_in_world = model_pose * volume_pose_;

	if (params_.model_k_means.k != 2 && params_.model_k_means.k != 6) {
		cout << "only 2 or 6 means supported" << endl;
		exit(1);
	}

	// you keep needing these values
	const size_t cell_count = buffer_mean_list_[0]->getSizeInCells();

#if 0
	// this was the old broken way:
	kernel_add_frame_to_2_means_.runKernel(*buffer_mean_list_[0], *buffer_count_list_[0], *buffer_mean_list_[1], *buffer_count_list_[1],
		frame.image_buffer_depth, frame.image_buffer_segments, 0,
		params_.volume.cell_size, volume_pose_in_world, params_.camera.focal, params_.camera.center);
#endif


	// this is the new way which should work:
#if 0
	kernel_add_frame_to_2_means_using_normals_.runKernel(*buffer_mean_list_[0], *buffer_count_list_[0], *buffer_mean_list_[1], *buffer_count_list_[1],
		frame.image_buffer_depth, frame.image_buffer_normals, frame.image_buffer_segments, 0,
		params_.volume.cell_size, volume_pose_in_world, params_.camera.focal, params_.camera.center);
#endif

	if (params_.model_k_means.store_normals) {
		// this is the memory intensive "store" way:
		KernelAddFrameTo2MeansUsingStoredNormals _KernelAddFrameTo2MeansUsingStoredNormals(*all_kernels_);
		_KernelAddFrameTo2MeansUsingStoredNormals.runKernel(
			*buffer_mean_list_[0], *buffer_count_list_[0], *buffer_normal_list_[0],
			*buffer_mean_list_[1], *buffer_count_list_[1],*buffer_normal_list_[1],
			frame.image_buffer_depth, frame.image_buffer_normals, frame.image_buffer_segments, 0,
			params_.volume.cell_size, volume_pose_in_world, params_.camera.focal, params_.camera.center, params_.volume.min_truncation_distance);
	}
	else if (params_.model_k_means.k == 6) {
		float min_dot_product = cos(params_.model_k_means.compatibility_add_max_angle_degrees * M_PI / 180);

		// this implies the fixed volume idea...
		for (size_t i = 0; i < params_.model_k_means.k; ++i) {
			// add to kernels based on compatibility with reference normal?
			KernelAddFrameIfCompatible _KernelAddFrameIfCompatible(*all_kernels_);
			_KernelAddFrameIfCompatible.runKernel(*buffer_mean_list_[i], *buffer_count_list_[i],
				frame.image_buffer_depth, frame.image_buffer_normals, frame.image_buffer_segments, 0,
				params_.volume.cell_size, volume_pose_in_world, params_.camera.focal, params_.camera.center,
				fixed_normal_list_[i], min_dot_product, params_.model_k_means.empty_always_included, params_.model_k_means.cos_weight, params_.volume.min_truncation_distance);
		}
	}

	///// now debug
	const int debug = true;
	if (debug) {

		if (update_interface_) {
			// get points with any weight for sanity
			//void VolumeBuffer::getNonzeroPointsAndFloatValues(const Eigen::Affine3f& pose, float cell_size, float epsilon, std::vector<std::pair<Eigen::Vector3f, float> >& result)
			//void convertPointsAndFloatsToMeshVertices(std::vector<std::pair<Eigen::Vector3f, float> > const& points_and_d, MeshVertexVector & result);
			if (false){
				std::vector<std::pair<Eigen::Vector3f, float> > points_and_values;
				buffer_count_list_[0]->getNonzeroPointsAndFloatValues(volume_pose_in_world, params_.volume.cell_size, 1e-6, points_and_values);
				MeshVertexVectorPtr vertices (new MeshVertexVector);
				convertPointsAndFloatsToMeshVertices(points_and_values, Eigen::Array4ub(0,0,255,0), *vertices );
				update_interface_->updatePointCloud("buffer_count_list_0", vertices);
			}

			if (false){
				std::vector<std::pair<Eigen::Vector3f, float> > points_and_values;
				buffer_count_list_[1]->getNonzeroPointsAndFloatValues(volume_pose_in_world, params_.volume.cell_size, 1e-6, points_and_values);
				MeshVertexVectorPtr vertices (new MeshVertexVector);
				convertPointsAndFloatsToMeshVertices(points_and_values, Eigen::Array4ub(0,255,0,0), *vertices );
				update_interface_->updatePointCloud("buffer_count_list_1", vertices);
			}

			// look at stupid dot products
			if (false){
				KernelDotVolumeNormal _KernelDotVolumeNormal(*all_kernels_);

				VolumeBuffer dot_products_0(all_kernels_);
				Eigen::Vector3f vector(5,5,5);
				_KernelDotVolumeNormal.runKernel(*buffer_count_list_[0], dot_products_0, vector);

				std::vector<std::pair<Eigen::Vector3f, float> > points_and_values;
				dot_products_0.getNonzeroPointsAndFloatValues(volume_pose_in_world, params_.volume.cell_size, 1e-6, points_and_values);
				MeshVertexVectorPtr vertices (new MeshVertexVector);
				convertPointsAndFloatsToMeshVertices(points_and_values, Eigen::Array4ub(0,255,0,0), *vertices );
				update_interface_->updatePointCloud("dot_products_0", vertices);
			}


			// look at normals
			KernelNormalsToColorImage _KernelNormalsToColorImage(*all_kernels_);
			KernelExtractFloat4ForPointImage _KernelExtractFloat4ForPointImage(*all_kernels_);
			KernelApplyPoseToNormals _KernelApplyPoseToNormals(*all_kernels_);
			KernelComputeNormalVolume _KernelComputeNormalVolume(*all_kernels_);
			KernelAddFrame _KernelAddFrame(*all_kernels_);

			// todo: compare frame normals to volume normals for each
			ImageBuffer normals_color(all_kernels_->getCL());
			ImageBuffer normals_from_volume(all_kernels_->getCL());
			{
				// make the frame normals in the global frame instead
				// need a copy because this happens in place
				//kernel_apply_pose_to_normals_.runKernel(normals_for_points.getBuffer(), volume_pose_in_world.inverse(), normals_for_points.getSizeBytes() / (4 * sizeof(float)));
				// blah...maybe not..

				_KernelNormalsToColorImage.runKernel(frame.image_buffer_normals, normals_color);
				imshow("frame.image_buffer_normals", normals_color.getMat());
			}

			// compare to the normals we get if we throw the frame in a fresh volume, and do volumetric normals...
			if (false) {
				VolumeBuffer volume_d_for_frame_only(all_kernels_, params_.volume.cell_count, sizeof(float));
				VolumeBuffer volume_dw_for_frame_only(all_kernels_, params_.volume.cell_count, sizeof(float));
				volume_d_for_frame_only.setFloat(-1);
				volume_dw_for_frame_only.setFloat(0);

				_KernelAddFrame.runKernel(volume_d_for_frame_only, volume_dw_for_frame_only,
					frame.image_buffer_depth, frame.image_buffer_segments, 0, params_.volume.cell_size,
					volume_pose_in_world, params_.camera.focal, params_.camera.center, params_.volume.min_truncation_distance);

				// then compute normals
				VolumeBuffer volume_normals(all_kernels_);
				_KernelComputeNormalVolume.runKernel(volume_d_for_frame_only, volume_normals);

				// then extract normals
				ImageBuffer normals_for_points(all_kernels_->getCL());
				_KernelExtractFloat4ForPointImage.runKernel(volume_normals, frame.image_buffer_points, normals_for_points, volume_pose_in_world, params_.volume.cell_size);

				// then shift normals to global frame?  No, they are in volume frame already...put in camera frame to compare to unshifted camera frame normals...
				_KernelApplyPoseToNormals.runKernel(normals_for_points.getBuffer(), volume_pose_in_world, normals_for_points.getSizeBytes() / (4 * sizeof(float)));

				// then show them...good grief
				ImageBuffer normals_for_points_image(all_kernels_->getCL());
				_KernelNormalsToColorImage.runKernel(normals_for_points, normals_for_points_image);
				imshow("normals_for_points_image", normals_for_points_image.getMat());
			}



			// could rewrite as loop for more k's
			if (params_.model_k_means.store_normals && params_.model_k_means.k == 2) {
				{
					_KernelExtractFloat4ForPointImage.runKernel(*buffer_normal_list_[0], frame.image_buffer_points, normals_from_volume, volume_pose_in_world, params_.volume.cell_size);
					//_KernelApplyPoseToNormals.runKernel(normals_from_volume_buffer, volume_pose_in_world, normals_from_volume.getSizeBytes() / (4 * sizeof(float)));
					_KernelNormalsToColorImage.runKernel(normals_from_volume, normals_color);
					imshow("buffer_normal_list_[0]", normals_color.getMat());
				}

				{
					_KernelExtractFloat4ForPointImage.runKernel(*buffer_normal_list_[1], frame.image_buffer_points, normals_from_volume, volume_pose_in_world, params_.volume.cell_size);
					_KernelNormalsToColorImage.runKernel(normals_from_volume, normals_color);
					//_KernelApplyPoseToNormals.runKernel(normals_from_volume_buffer, volume_pose_in_world, normals_from_volume.getSizeBytes() / (4 * sizeof(float)));
					imshow("buffer_normal_list_[1]", normals_color.getMat());
				}
			}
		}


#if 0
		// pick pixel here?
		// could put this part in model base for example?
		// or even within pick pixel itself if it were aware of depth
		// duplicated for kmeans
		if (pick_pixel_) {
			cv::Point2i pixel_cv = pick_pixel_->getPixel();
			Eigen::Array2i pixel_eigen(pixel_cv.x, pixel_cv.y);
			if ((pixel_eigen != last_pick_pixel_).any() && (pixel_eigen >= 0).all()) {
				last_pick_pixel_ = pixel_eigen;
				float depth_at_pick_pixel = frame.mat_depth.at<float>(last_pick_pixel_[1], last_pick_pixel_[0]);
				depth_at_pick_pixel -= params_.model_histogram.debug_pick_pixel_depth_offset;
				last_pick_pixel_world_point_ = model_pose.inverse() * EigenUtilities::depthToPoint(params_.camera.focal, params_.camera.center, last_pick_pixel_, depth_at_pick_pixel);
				last_pick_pixel_camera_point_ = model_pose.inverse() * Eigen::Vector3f(0,0,0);
			}
		}

		// this is duplicated all over as well, and now broken...fix sometime

		// get means along ray
		{
			// all points along debug ray
			{
				Eigen::Vector3f ray_to_point = (last_pick_pixel_world_point_ - last_pick_pixel_camera_point_);
				float t_for_point = ray_to_point.norm();
				ray_to_point /= t_for_point;

				float t_step = params_.volume.cell_size;

				MeshVertexVectorPtr mesh_vertices_on_ray(new MeshVertexVector);
				std::vector<cv::Mat> image_list;

				for (int step = -params_.model_histogram.debug_points_along_ray; step <= params_.model_histogram.debug_points_along_ray; ++step) {
					Eigen::Vector3f point_on_ray = last_pick_pixel_camera_point_+ (t_for_point + t_step * step) * ray_to_point;

					// add points to point cloud
					MeshVertex vertex;
					vertex.p.head<3>() = point_on_ray;
					vertex.p[3] = 1;
					vertex.c = Eigen::Array4ub(0,0,255,255);
					mesh_vertices_on_ray->push_back(vertex);

					// also get kmeans image
					std::vector<float> means, counts;
					getMeansAndCountsForPoint(point_on_ray, means, counts);

					int width = params_.model_histogram.mat_width_per_bin * params_.model_histogram.bin_count;
					int height = params_.model_histogram.mat_height;
					cv::Mat kmeans_mat = drawKMeansImage(means, counts, params_.model_histogram.min_value, params_.model_histogram.max_value, width, height, true, true);

					// add histogram image to list
					image_list.push_back(kmeans_mat);
				}

				if (update_interface_) update_interface_->updatePointCloud(debug_string_prefix_ + "mesh_vertices_on_ray", mesh_vertices_on_ray);

				// build and show image
				// this could still crash...
				if (!image_list.empty() && !image_list[0].empty()) {
					cv::Mat all_images = createMxN(image_list.size(), 1, image_list);
					cv::imshow(debug_string_prefix_ + "all_means", all_images);
				}
			}
		}
#endif


	} // if debug
}

void ModelKMeans::generateMesh(MeshVertexVector & vertex_list, TriangleVector & triangle_list)
{
	// you keep needing these values
	const size_t cell_count = buffer_mean_list_[0]->getSizeInCells();

	// considering computeMinAbsVolume as default mesh for now...

	VolumeBuffer combined_d(all_kernels_);
	VolumeBuffer combined_dw(all_kernels_);
	computeMinAbsVolume(params_.model_k_means.minimum_relative_count, combined_d, combined_dw);

	MeshPtr mesh = MeshUtilities::generateMesh(combined_d, combined_dw, params_.volume.cell_size, params_.model_k_means.default_mesh_color, volume_pose_);

	// wasteful copy
	vertex_list = mesh->vertices;
	triangle_list = mesh->triangles;
}

void ModelKMeans::generateMeshAndValidity(MeshVertexVector & vertex_list, TriangleVector & triangle_list, std::vector<bool> & vertex_validity, std::vector<bool> & triangle_validity)
{
	// todo?
}

void ModelKMeans::generateAllMeshes(std::vector<std::pair<std::string, MeshPtr> > & names_and_meshes)
{
	ModelBase::generateAllMeshes(names_and_meshes);

	VolumeBuffer combined_d(all_kernels_, params_.volume.cell_count, sizeof(float));
	VolumeBuffer combined_dw(all_kernels_, params_.volume.cell_count, sizeof(float));

	// do the overall weighted sum
	// sum kernel?
	{
		KernelAddFloatsWithWeights _KernelAddFloatsWithWeights(*all_kernels_);
		combined_d.setFloat(-1);
		combined_dw.setFloat(0);
		for (size_t i = 0; i < params_.model_k_means.k; ++i) {
			_KernelAddFloatsWithWeights.runKernel(combined_d.getBuffer(), combined_dw.getBuffer(), buffer_mean_list_[i]->getBuffer(), buffer_count_list_[i]->getBuffer(), combined_d.getSizeInCells());
		}

		MeshPtr mesh = MeshUtilities::generateMesh(combined_d, combined_dw, params_.volume.cell_size, params_.model_k_means.default_mesh_color, volume_pose_);
		names_and_meshes.push_back(std::make_pair("mesh_sum_all", mesh));
	}

	std::vector<MeshPtr> individual_meshes = getPartialMeshes();
	for (size_t i = 0; i < buffer_mean_list_.size(); ++i) {
		std::string name = (boost::format("mesh_%d") % i).str();
		names_and_meshes.push_back(std::make_pair(name, individual_meshes[i]));
	}

    // by normals (seems to be working)
    {
        boost::shared_ptr<std::vector<MeshPtr> > invalid (new std::vector<MeshPtr>); // debug only

        std::vector<MeshPtr> individual_meshes_normal = getValidPartialMeshesByNormal(individual_meshes, invalid);
        for (size_t i = 0; i < individual_meshes_normal.size(); ++i) {
            std::string name = (boost::format("mesh_normals_%d") % i).str();
            names_and_meshes.push_back(std::make_pair(name, individual_meshes_normal[i]));
        }

        // more debug
        for (size_t i = 0; i < invalid->size(); ++i) {
            std::string name = (boost::format("mesh_normals_invalid_%d") % i).str();
            names_and_meshes.push_back(std::make_pair(name, invalid->at(i)));
        }
    }

    // by empty space violation (not working)
    if (false){
        boost::shared_ptr<std::vector<MeshPtr> > invalid (new std::vector<MeshPtr>); // debug only

        std::vector<MeshPtr> individual_meshes_violate = getValidPartialMeshesByEmptyViolation(individual_meshes, invalid);
        for (size_t i = 0; i < buffer_mean_list_.size(); ++i) {
            std::string name = (boost::format("mesh_violate_%d") % i).str();
            names_and_meshes.push_back(std::make_pair(name, individual_meshes_violate[i]));
        }

        // more debug
        for (size_t i = 0; i < invalid->size(); ++i) {
            std::string name = (boost::format("mesh_violate_invalid_%d") % i).str();
            names_and_meshes.push_back(std::make_pair(name, invalid->at(i)));
        }
    }


	// also walk variations in min_fraction param
    if (false) {
        const float increment = 0.2;
        for (float min_fraction = 0; min_fraction < 1; min_fraction += increment) {
            computeMinAbsVolume(min_fraction, combined_d, combined_dw);
            MeshPtr mesh = MeshUtilities::generateMesh(combined_d, combined_dw, params_.volume.cell_size, params_.model_k_means.default_mesh_color, volume_pose_);
            std::string name = (boost::format("mesh_min_fraction_%d") % min_fraction).str();
            names_and_meshes.push_back(std::make_pair(name, mesh));
        }
    }
}

void ModelKMeans::deallocateBuffers()
{
	// todo?
}

void ModelKMeans::save(fs::path const& folder)
{
	ModelBase::save(folder);

	// more?
}

void ModelKMeans::load(fs::path const& folder)
{
	ModelBase::load(folder);

	// more?
}


void ModelKMeans::getBoundingLines(MeshVertexVector & vertex_list)
{
	Eigen::Vector4ub color(0,0,255,0);
	::getBoundingLines(buffer_mean_list_[0]->getVolumeCellCounts(), params_.volume.cell_size, volume_pose_, color, vertex_list);
}


void ModelKMeans::refreshUpdateInterface()
{
	if (update_interface_) {
		{
			MeshVertexVectorPtr vertices_ptr (new MeshVertexVector);
			getBoundingLines(*vertices_ptr);
			update_interface_->updateLines("ModelKMeans", vertices_ptr);
		}

		// fixed normals (may not matter depending on how run
		if (false) {
			MeshVertexVectorPtr vertices_ptr_small (new MeshVertexVector);
			MeshVertexVectorPtr vertices_ptr_medium (new MeshVertexVector);
			MeshVertexVectorPtr vertices_ptr_large (new MeshVertexVector);
			for (size_t i = 0; i < fixed_normal_list_.size(); ++i) {
				MeshVertex v1, v2;
				v1.p = Eigen::Vector4f(0,0,0,1);
				v2.p.head<3>() = fixed_normal_list_[i] * params_.model_k_means.fixed_normal_view_length;
				v2.p[3] = 1;
				v1.n = v2.n = Eigen::Vector4f::Zero();
				v1.c = v2.c = fixed_color_list_[i];
				vertices_ptr_small->push_back(v1);
				vertices_ptr_small->push_back(v2);

				// sometimes big as well
				if (!debug_compatible_last_frame_.empty() && debug_compatible_last_frame_[i]) {
					vertices_ptr_medium->push_back(v1);
					vertices_ptr_medium->push_back(v2);
				}

				// sometimes very big
				if (!debug_best_last_frame_.empty() && debug_best_last_frame_[i]) {
					vertices_ptr_large->push_back(v1);
					vertices_ptr_large->push_back(v2);
				}

			}
			update_interface_->updateLines("Fixed normals small", vertices_ptr_small);
			const static std::string name_medium = "Fixed normals medium";
			update_interface_->updateLines(name_medium, vertices_ptr_medium);
			update_interface_->updateScale(name_medium, 5);
			const static std::string name_big = "Fixed normals large";
			update_interface_->updateLines(name_big, vertices_ptr_large);
			update_interface_->updateScale(name_big, 10);
		}

		// also slice?
		{
			int axis = params_.model_k_means.slice_axis;
			int position = trackbar_window_.getTrackbarValue();
			MeshVertexVectorPtr vertices_ptr (new MeshVertexVector);
			getSliceLines(params_.volume.cell_count, params_.volume.cell_size, volume_pose_, axis, position, Eigen::Array4ub(0,0,0,0), *vertices_ptr);
			update_interface_->updateLines("Slice", vertices_ptr);
		}
	}
}

bool ModelKMeans::getMeansAndCountsForVoxel(const Eigen::Array3i & voxel, std::vector<float> & result_means, std::vector<float> & result_counts)
{
	result_means.assign(params_.model_k_means.k, 0);
	result_counts.assign(params_.model_k_means.k, 0);
	if (!isVertexInVolume(buffer_mean_list_[0]->getVolumeCellCounts(), voxel)) return false;

	KernelExtractVolumeFloat _KernelExtractVolumeFloat(*all_kernels_);

	for (size_t cluster = 0; cluster < params_.model_k_means.k; ++cluster) {
		float value_mean;
		_KernelExtractVolumeFloat.runKernel(*buffer_mean_list_[cluster], voxel, value_mean);
		float value_count;
		_KernelExtractVolumeFloat.runKernel(*buffer_count_list_[cluster], voxel, value_count);

		result_means[cluster] = value_mean;
		result_counts[cluster] = value_count;
	}

	return true;
}

bool ModelKMeans::getMeansAndCountsForPoint(const Eigen::Vector3f & world_point, std::vector<float> & result_means, std::vector<float> & result_counts)
{
	Eigen::Vector3f point_in_volume = volume_pose_.inverse() * world_point;

	// this may put slightly too negative points into the voxel grid...ok for now...do correct rounding at some point
	Eigen::Array3i nearest_voxel = EigenUtilities::roundPositiveVector3fToInt((point_in_volume / params_.volume.cell_size)).array();

	return getMeansAndCountsForVoxel(nearest_voxel, result_means, result_counts);
}

void ModelKMeans::computeMinAbsVolume(float minimum_weight_fraction, VolumeBuffer & combined_mean, VolumeBuffer & combined_weight)
{
	KernelMaxFloats _KernelMaxFloats(*all_kernels_);

	// first need max weight
	VolumeBuffer max_weight(all_kernels_, params_.volume.cell_count, sizeof(float));
	VolumeBuffer max_index(all_kernels_, params_.volume.cell_count, sizeof(int));
	max_weight.setFloat(0);
	max_index.setInt(-1);

	for (size_t i = 0; i < params_.model_k_means.k; ++i) {
		_KernelMaxFloats.runKernel(max_weight.getBuffer(), max_index.getBuffer(), buffer_mean_list_[i]->getBuffer(), (int)i, max_weight.getSizeInCells());
	}

	// whee...now have max weight...
	KernelMinAbsFloatsWithWeightsAndMinimumWeightFraction _KernelMinAbsFloatsWithWeightsAndMinimumWeightFraction(*all_kernels_);

	combined_mean.resize(params_.volume.cell_count, sizeof(float));
	combined_weight.resize(params_.volume.cell_count, sizeof(float));
	combined_mean.setFloat(-1);
	combined_weight.setFloat(0);

	for (size_t i = 0; i < params_.model_k_means.k; ++i) {
		_KernelMinAbsFloatsWithWeightsAndMinimumWeightFraction.runKernel(
			combined_mean.getBuffer(), combined_weight.getBuffer(),
			buffer_mean_list_[i]->getBuffer(), buffer_count_list_[i]->getBuffer(),
			max_weight.getBuffer(), minimum_weight_fraction, combined_mean.getSizeInCells());
	}
}

std::vector<MeshPtr> ModelKMeans::getPartialMeshes()
{
	std::vector<MeshPtr> result;
	for (size_t i = 0; i < buffer_mean_list_.size(); ++i) {
		MeshPtr mesh = MeshUtilities::generateMesh(*buffer_mean_list_[i], *buffer_count_list_[i], params_.volume.cell_size, fixed_color_list_[i], volume_pose_);
		result.push_back(mesh);
	}
	return result;
}

std::vector<MeshPtr> ModelKMeans::getValidPartialMeshesByNormal(const std::vector<MeshPtr> & mesh_list, boost::shared_ptr<std::vector<MeshPtr> > result_also_invalid)
{
	std::vector<MeshPtr> result;
	for (size_t i = 0; i < mesh_list.size(); ++i) {
		const MeshPtr & mesh = mesh_list[i];
		// loop over mesh vertices, remove those which violate normal of volume they are in
		float min_dot_product = cos(params_.model_k_means.compatibility_add_max_angle_degrees * M_PI / 180);
		std::vector<bool> vec_valid_bool(mesh->vertices.size(), true);
		for (size_t p = 0; p < mesh->vertices.size(); ++p) {
			Eigen::Vector4f & normal = mesh->vertices[p].n;
			float normal_dot = normal.head<3>().dot(fixed_normal_list_[i]);
			if (normal_dot < min_dot_product) vec_valid_bool[p] = false;
		}

        MeshPtr mesh_valid_only(new Mesh);
        MeshUtilities::extractVerticesAndTriangles(*mesh, vec_valid_bool, true, *mesh_valid_only);
        result.push_back(mesh_valid_only);

        if (result_also_invalid) {
            MeshPtr mesh_invalid_only(new Mesh);
            MeshUtilities::extractVerticesAndTriangles(*mesh, vec_valid_bool, false, *mesh_invalid_only);
            result_also_invalid->push_back(mesh_invalid_only);
        }
	}
	return result;
}

std::vector<MeshPtr> ModelKMeans::getValidPartialMeshesByEmptyViolation(const std::vector<MeshPtr> & mesh_list, boost::shared_ptr<std::vector<MeshPtr> > result_also_invalid)
{
	std::vector<MeshPtr> result;

	KernelSetUChar _KernelSetUChar(*all_kernels_);
	KernelMarkPointsViolateEmpty _KernelMarkPointsViolateEmpty(*all_kernels_);

	for (size_t i = 0; i < mesh_list.size(); ++i) {
		const MeshPtr & mesh = mesh_list[i];

        // put these out here, so they can get mushed onto the result empty if needed
        MeshPtr mesh_valid_only(new Mesh);
        MeshPtr mesh_invalid_only(new Mesh);

        if (!mesh->triangles.empty()) {

            // can now test removal of additional empty space violating vertices
            BufferWrapper buffer_points(all_kernels_->getCL());
            // get a vector of floats for the points (wasteful..)
            std::vector<float> vec_points(mesh->vertices.size() * 4);
            for (size_t p = 0; p < mesh->vertices.size(); ++p) {
                Eigen::Vector4f & point = mesh->vertices[p].p;
                vec_points[4 * p] = point[0];
                vec_points[4 * p + 1] = point[1];
                vec_points[4 * p + 2] = point[2];
                vec_points[4 * p + 3] = point[3];
            }
            buffer_points.writeFromFloatVector(vec_points);

            BufferWrapper buffer_invalid(all_kernels_->getCL());
            buffer_invalid.reallocate(mesh->vertices.size() * sizeof(unsigned char));
            _KernelSetUChar.runKernel(buffer_invalid.getBuffer(), buffer_invalid.getBufferSize() / sizeof(unsigned char), false);

            for (size_t j = 0; j < buffer_mean_list_.size(); ++j) {
                if (j == i) continue;

                // more param here?
                //const float min_value_invalid = 3 * params_.volume.cell_size;
                const float min_value_invalid = 10 * params_.volume.cell_size; // should do nothing

                _KernelMarkPointsViolateEmpty.runKernel(*buffer_mean_list_[j], *buffer_count_list_[j],
                                                        buffer_points.getBuffer(), buffer_invalid.getBuffer(), mesh->vertices.size(),
                                                        params_.volume.cell_size, volume_pose_, min_value_invalid);
            }

            // should now have "true" for violation of any other volume
            // take a look at the violating points...
            std::vector<unsigned char> vec_invalid_char(buffer_invalid.getBufferSize() / sizeof(unsigned char));
            buffer_invalid.readToByteVector(vec_invalid_char);
            // this is silly, but at least it's type-safe
            std::vector<bool> vec_invalid_bool(vec_invalid_char.begin(), vec_invalid_char.end());
            // this is also silly
            std::vector<bool> vec_valid_bool;
            vec_valid_bool.reserve(vec_invalid_bool.size());
            BOOST_FOREACH(const bool & b, vec_invalid_bool) {
                vec_valid_bool.push_back(!b);
            }

            MeshUtilities::extractVerticesAndTriangles(*mesh, vec_valid_bool, true, *mesh_valid_only);

            if (result_also_invalid) {
                MeshUtilities::extractVerticesAndTriangles(*mesh, vec_valid_bool, false, *mesh_invalid_only);
            }
        } // !empty

        result.push_back(mesh_valid_only);
         if (result_also_invalid) {
             result_also_invalid->push_back(mesh_invalid_only);
         }
    }

	return result;
}
