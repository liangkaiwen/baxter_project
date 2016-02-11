#include "model_patch.h"

#include "Noise.h"
#include "disjoint_set.h"
#include "image_to_cloud.h"

#include "util.h"

bool operator<(const MergeEdge &a, const MergeEdge &b) {
	return a.w < b.w;
}

ModelPatch::ModelPatch(boost::shared_ptr<OpenCLAllKernels> all_kernels, VolumeModelerAllParams const& params, boost::shared_ptr<Alignment> alignment_ptr, RenderBuffers const& render_buffers)
	: ModelGrid(all_kernels, params, alignment_ptr, render_buffers)
{
}

ModelPatch::ModelPatch(ModelPatch const& other)
	: ModelGrid(other)
{
}

ModelPatch* ModelPatch::clone()
{
	return new ModelPatch(*this);
}

void ModelPatch::updateModel(
	Frame & frame,
	const Eigen::Affine3f & model_pose)
{
	// note, not ModelGrid!!
	ModelBase::updateModel(frame, model_pose);

	cv::Mat render_segments, consistent_segments, segments;
	std::map<int,int> segment_sizes_map;
	std::map<int,Eigen::Vector3f> segment_normals_map;
	getSegmentation(frame, model_pose, render_segments, consistent_segments, segments, segment_sizes_map, segment_normals_map);

	debug_images_["render_segments"] = getColorSegmentMat(render_segments);
	debug_images_["consistent_segments"] = getColorSegmentMat(consistent_segments);
	debug_images_["segments"] = getColorSegmentMat(segments);

	// get the segmentation onto the gpu
	frame.mat_segments = segments;
	frame.copySegmentsToImageBuffer(); // replaces current contents

	// get the full cloud of points in the frame
	// todo: could just get the points from the frame (openCL)
	Eigen::Matrix4Xf cloud_4;
	imageToCloud(params_.camera, frame.mat_depth, cloud_4);

	Eigen::Affine3f camera_pose = model_pose.inverse();

	cout << "grid_list_.size() before updateModel: " << grid_list_.size() << endl;

	grids_updated_last_call_.clear();

	// increase age of all
	for (size_t i = 0; i < grid_list_.size(); ++i) {
		grid_list_[i]->age++;
	}

	// add points to volumes
	for (std::map<int,int>::const_iterator iter = segment_sizes_map.begin(); iter != segment_sizes_map.end(); ++iter) {
		int segment_id = iter->first;
		int volume_index = segment_id - 1;
		if (volume_index < 0) throw std::runtime_error("you suck at coding");

		cout << "processing segment_id: " << segment_id << endl;

		// get the subcloud for this segment_id
		Eigen::Matrix3Xf segment_cloud_in_frame (3, iter->second);
		int i_out = 0;
		for (int i_in = 0; i_in < segments.total(); ++i_in) {
			if (segments.at<int>(i_in) == segment_id) {
				segment_cloud_in_frame.col(i_out++) = cloud_4.col(i_in).head<3>();
			}
		}

		if (i_out != iter->second) throw std::runtime_error("size mismatch");

		Eigen::Matrix3Xf segment_cloud_in_world = camera_pose * segment_cloud_in_frame;

		if (volume_index < grid_list_.size()) {
			// existing
			cout << "updating existing patch: " << volume_index << endl;
			GridStruct & grid_struct = *grid_list_[volume_index];

			// see if expand necessary

			// transform segment cloud to tsdf
			// note, this is not "centered" as for creation, but in the TSDF
			Eigen::Matrix3Xf segment_cloud_in_tsdf = grid_struct.getExternalTSDFPose().inverse() * segment_cloud_in_world;

			// check against bounding box
			Eigen::Array3f bb_cloud_min = segment_cloud_in_tsdf.rowwise().minCoeff();
			Eigen::Array3f bb_cloud_max = segment_cloud_in_tsdf.rowwise().maxCoeff();
			Eigen::Array3f bb_tsdf_min, bb_tsdf_max;
			grid_struct.tsdf_ptr->getAABB(bb_tsdf_min, bb_tsdf_max);

			if ( (bb_cloud_min < bb_tsdf_min).any() || (bb_cloud_max > bb_tsdf_max).any() ) {
				// must expand

				// assumes that coefficient-wise < produces 1's and 0's
				Eigen::Array3f bb_both_min = bb_cloud_min.min(bb_tsdf_min) - (bb_cloud_min < bb_tsdf_min).cast<float>() * params_.patch.border_expand;
				Eigen::Array3f bb_both_max = bb_cloud_max.max(bb_tsdf_max) + (bb_cloud_max > bb_tsdf_max).cast<float>() * params_.patch.border_expand;

				Eigen::Array3f bb_both_min_cells_f = bb_both_min / params_.volume.cell_size - 1;
				Eigen::Array3f bb_both_max_cells_f = bb_both_max / params_.volume.cell_size + 1;
				Eigen::Array3i bb_both_min_cells( (int)bb_both_min_cells_f[0], (int)bb_both_min_cells_f[1], (int)bb_both_min_cells_f[2] );
				Eigen::Array3i bb_both_max_cells( (int)bb_both_max_cells_f[0], (int)bb_both_max_cells_f[1], (int)bb_both_max_cells_f[2] );
				Eigen::Array3i both_cell_counts = bb_both_max_cells - bb_both_min_cells;

				// create new larger tsdf
                OpenCLTSDFPtr new_tsdf_ptr(new OpenCLTSDF(all_kernels_,
					params_.camera.focal.x(), params_.camera.focal.y(),
					params_.camera.center.x(), params_.camera.center.y(),
					params_.camera.size.x(), params_.camera.size.y(),
					both_cell_counts[0], both_cell_counts[1], both_cell_counts[2], params_.volume.cell_size,
                    params_.volume.max_weight_icp, params_.volume.max_weight_color, params_.volume.use_most_recent_color, params_.volume.min_truncation_distance, params_.grid.temp_folder));

				// merge into this tsdf
				// todo: exploit axis-aligned for speed (and maybe accuracy)
				// note that this is POSITIVE
				Eigen::Vector3f center_offset_copied_from_create = -bb_both_min_cells.cast<float>() * params_.volume.cell_size;
				Eigen::Affine3f center_offset_pose;
				center_offset_pose = Eigen::Translation3f(center_offset_copied_from_create);

#if 0
				// this should also work, although slowly
				new_tsdf_ptr->addVolume(*grid_struct.tsdf_ptr, center_offset_pose.inverse());
#endif
				new_tsdf_ptr->copyVolumeAxisAligned(*grid_struct.tsdf_ptr, Eigen::Array3i(0,0,0), -bb_both_min_cells);

				// update tsdf and centering pose
				grid_struct.tsdf_ptr = new_tsdf_ptr;
				grid_struct.pose_tsdf = grid_struct.pose_tsdf * center_offset_pose.inverse();
			}

			// add to this grid struct
			grid_struct.tsdf_ptr->addFrame(model_pose * grid_struct.getExternalTSDFPose(), 
				frame.image_buffer_depth,
				frame.image_buffer_color, 
				frame.image_buffer_segments, 
				frame.image_buffer_add_depth_weights, 
				frame.image_buffer_add_color_weights, 
				segment_id);
			grids_updated_last_call_.push_back(volume_index);
			grid_struct.age = 0;

			///////////////////
			// here we can re-align an existing volume
			// current thinking: should do this after adding to the volume, because volume was expanded based on new points
			if (params_.patch.test_patch_reorientation) {
				Eigen::Affine3f pose_for_original_points = grid_struct.pose_tsdf;
				std::vector<std::pair<Eigen::Vector3f, float> > points_and_d_in_old_tsdf;
				grid_struct.tsdf_ptr->getNonzeroFilteredVoxelCenters(pose_for_original_points, -2*params_.volume.cell_size, 2*params_.volume.cell_size, points_and_d_in_old_tsdf);

				// for now, just display
				if (update_interface_ && params_.patch.test_patch_reorientation_glfw) {
					MeshVertexVectorPtr vertices_to_show(new MeshVertexVector);
					convertPointsAndFloatsToMeshVertices(points_and_d_in_old_tsdf, *vertices_to_show);
					update_interface_->updatePointCloud("test_patch_reorientation", vertices_to_show);

					// show tsdf
					{
						MeshVertexVectorPtr vertices(new MeshVertexVector);
						Eigen::Vector4ub c (200,200,200,0);
						grid_struct.tsdf_ptr->getBoundingLines(pose_for_original_points, c, *vertices);
						update_interface_->updateLines("test_patch_reorientation tsdf", vertices);
					}

					cout << "test_patch_reorientation paused before recenter..." << endl;
					cv::waitKey();
				}

				// convert point format (sigh)
				Eigen::Matrix3Xf tsdf_points(3, points_and_d_in_old_tsdf.size());
				for (int i = 0; i < points_and_d_in_old_tsdf.size(); ++i) {
					tsdf_points.col(i) = points_and_d_in_old_tsdf[i].first;
				}

				// orientation for these points
				Eigen::Vector3f cloud_mean_ignore;
				Eigen::Affine3f rotate_cloud_to_axes;
				getCovarianceRotation(tsdf_points, cloud_mean_ignore, rotate_cloud_to_axes);

#if 0
				// any arbtrary wrong transformation should still work here..and seems to...
				cout << "INJECTING A TEST" << endl;
				rotate_cloud_to_axes = rotate_cloud_to_axes * Eigen::AngleAxisf(M_PI/4, Eigen::Vector3f(1,0,0));
#endif

				Eigen::Matrix3Xf cloud_rotated = rotate_cloud_to_axes * tsdf_points;

				// get tsdf values
				Eigen::Array3i cell_counts;
				Eigen::Vector3f center_offset;
				getRequiredTSDFValues(cloud_rotated, cell_counts, center_offset);

				Eigen::Affine3f new_pose_tsdf = rotate_cloud_to_axes.inverse() * Eigen::Translation3f(-center_offset);
				OpenCLTSDFPtr new_tsdf_ptr = createTSDF(cell_counts);

				//Eigen::Affine3f relative_pose_for_add_volume = grid_struct.pose_tsdf * new_pose_tsdf.inverse();
				Eigen::Affine3f relative_pose_for_add_volume = grid_struct.pose_tsdf.inverse() * new_pose_tsdf;
				new_tsdf_ptr->addVolume(*grid_struct.tsdf_ptr, relative_pose_for_add_volume);
				grid_struct.tsdf_ptr = new_tsdf_ptr;
				grid_struct.pose_tsdf = new_pose_tsdf;

				// debug the new tsdf
				if (update_interface_ && params_.patch.test_patch_reorientation_glfw) {
					Eigen::Affine3f pose_for_debug_after = new_pose_tsdf;
					std::vector<std::pair<Eigen::Vector3f, float> > points_and_d_new;
					grid_struct.tsdf_ptr->getNonzeroFilteredVoxelCenters(pose_for_debug_after, -2*params_.volume.cell_size, 2*params_.volume.cell_size, points_and_d_new);

					// new stuff
					MeshVertexVectorPtr vertices_to_show(new MeshVertexVector);
					convertPointsAndFloatsToMeshVertices(points_and_d_new, *vertices_to_show);
					update_interface_->updatePointCloud("test_patch_reorientation", vertices_to_show);

					// show tsdf
					{
						MeshVertexVectorPtr vertices(new MeshVertexVector);
						Eigen::Vector4ub c (200,200,200,0);
						grid_struct.tsdf_ptr->getBoundingLines(pose_for_debug_after, c, *vertices);
						update_interface_->updateLines("test_patch_reorientation tsdf", vertices);
					}

					cout << "test_patch_reorientation after recenter paused..." << endl;
					cv::waitKey();
				}


			}

		}
		else {
			// new 
			cout << "creating new patch for SUPPOSED (from segmentation) volume_index: " << volume_index << endl;

			if (!params_.patch.segments_create_of_all_sizes && iter->second < params_.patch.segments_min_size) {
				cout << "Skipping creation of small segment of size: " << iter->second << endl;
				continue;
			}

			Eigen::Vector3f cloud_mean;
			Eigen::Affine3f rotate_cloud_to_axes;
			Eigen::Array3i cell_counts;
			Eigen::Vector3f center_offset;
			getCenteringPose(segment_cloud_in_world, cloud_mean, rotate_cloud_to_axes, cell_counts, center_offset);

			// keep rotation as part of the tsdf bit, not the external bit
			Eigen::Affine3f pose_external, pose_tsdf;
			pose_external = Eigen::Translation3f(cloud_mean);
			pose_tsdf = rotate_cloud_to_axes.inverse() * Eigen::Translation3f(-center_offset);
			appendNewGridStruct(pose_external, pose_tsdf, cell_counts);


			GridStruct & grid_struct = *grid_list_.back();
			grid_struct.tsdf_ptr->addFrame(model_pose * grid_struct.getExternalTSDFPose(), 
				frame.image_buffer_depth,
				frame.image_buffer_color, 
				frame.image_buffer_segments, 
				frame.image_buffer_add_depth_weights, 
				frame.image_buffer_add_color_weights, 
				segment_id);
			grids_updated_last_call_.push_back(grid_list_.size() - 1);
			grid_struct.age = 0; // redundant for new volume


			///////// debug stuff in 3D viewer
			if (params_.patch.debug_patch_creation && update_interface_) {
				{
					// segment_cloud_in_world
					MeshVertexVectorPtr vertices(new MeshVertexVector);
					for (int i = 0; i < segment_cloud_in_world.cols(); ++i) {
						vertices->push_back(MeshVertex());
						MeshVertex & v = vertices->back();
						v.p.head<3>() = segment_cloud_in_world.col(i);
						v.c = Eigen::Vector4ub(255,255,255,0);
					}
					update_interface_->updatePointCloud("debug segment_cloud_in_world", vertices);
				}

				{
					// tsdf from last in grid list (in world now with "pose")
					MeshVertexVectorPtr vertices(new MeshVertexVector);
					Eigen::Vector4ub c (200,200,200,0);
					grid_list_.back()->tsdf_ptr->getBoundingLines(model_pose * grid_list_.back()->getExternalTSDFPose(), c, *vertices);
					update_interface_->updateLines("debug tsdf", vertices);
				}

#if 0
				{
					// eigenvectors
					MeshVertexVectorPtr vertices(new MeshVertexVector);
					for (int i = 0; i < 3; ++i) {
						Eigen::Vector4f p1, p2;
						p1.head<3>() = mean;
						p2.head<3>() = mean + solver.eigenvectors().col(i) * sqrt(fabs(solver.eigenvalues()[i]));
						Eigen::Vector4ub c(0,0,255,0);
						OpenCLTSDF::appendLine(*vertices, p1, p2, c);
					}
					update_interface_->updateLines("eigenvectors", vertices);
				}

				if (false) {
					// reference block-centered axes
					MeshVertexVectorPtr vertices(new MeshVertexVector);
					for (int i = 0; i < 3; ++i) {
						Eigen::Vector4f p1, p2;
						p1.head<3>() = mean;
						Eigen::Vector3f offset (0,0,0);
						offset[i] = 0.1;
						p2.head<3>() = mean + offset;
						Eigen::Vector4ub c(0,255,0,0);
						OpenCLTSDF::appendLine(*vertices, p1, p2, c);
					}
					update_interface_->updateLines("debug axes", vertices);
				}

				if (false) {
					// rotated eigenvectors...
					MeshVertexVectorPtr vertices(new MeshVertexVector);
					Eigen::Matrix3f ev = solver.eigenvectors();
					Eigen::Affine3f eigen_rotate;
					eigen_rotate = ev;
					eigen_rotate = eigen_rotate.inverse(); // try inverse: yup, that aligns with world axes
					Eigen::Matrix3f rotated_vectors = eigen_rotate * ev;
					for (int i = 0; i < 3; ++i) {
						Eigen::Vector4f p1, p2;
						p1.head<3>() = mean;
						p2.head<3>() = mean + rotated_vectors.col(i);
						Eigen::Vector4ub c(255,255,0,0);
						OpenCLTSDF::appendLine(*vertices, p1, p2, c);
					}
					update_interface_->updateLines("rotated axes", vertices);
				}
#endif

				{
					MeshVertexVectorPtr vertices(new MeshVertexVector);
					grid_struct.tsdf_ptr->getPrettyVoxelCenters(model_pose * grid_struct.getExternalTSDFPose(), *vertices);
					update_interface_->updatePointCloud("debug getPrettyVoxelCenters", vertices);
				}

				cout << "debug pause" << endl;
				cv::waitKey();

			} // update interface debugging

		} // else (new grid struct)
	} // loop over segments

	bool changed_keyframe_ignore = UpdateKeyframeAndVolumeGraph(frame);
}


void ModelPatch::getCovarianceRotation(Eigen::Matrix3Xf const& cloud_in_world, 
	Eigen::Vector3f & result_cloud_mean, 
	Eigen::Affine3f & result_rotate_cloud_to_axes)
{
	result_cloud_mean = cloud_in_world.rowwise().mean();

	// originally I wrote this for debugging, but it's actually faster than my eigen way below...
	Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
	for (int i = 0; i < cloud_in_world.cols(); ++i) {
		Eigen::Vector3f p = cloud_in_world.col(i) - result_cloud_mean;
		cov(0,0) += p[0] * p[0];
		cov(1,0) += p[1] * p[0];
		cov(1,1) += p[1] * p[1];
		cov(2,0) += p[2] * p[0];
		cov(2,1) += p[2] * p[1];
		cov(2,2) += p[2] * p[2];
	}
	cov(0,1) = cov(1,0);
	cov(0,2) = cov(2,0);
	cov(1,2) = cov(2,1);
	cov = cov / cloud_in_world.cols();

	////////////////////////////
	// try to figure out a fast eigen way to do it
	// so yeah...this looks correct now, but is SLOWER THAN MY WAY!
	// todo: use Matrix4Xf instead for SIMD
#if 0
	{
		boost::timer t;
		Eigen::Matrix3Xf centered2 = segment_cloud.colwise() - mean;
		Eigen::Matrix3f cov = centered2 * centered2.transpose();
		cov = cov / segment_cloud.cols();
		cout << "cov better:\n" << cov << endl;
		cout << "time for eigen way: " << t.elapsed() << endl;
	}
#endif

	// here we actually do the eigenvalues
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
	if (solver.info() != Eigen::Success) throw std::runtime_error("SelfAdjointEigenSolver problem");

	// rotate cloud by eigenvectors inverse
	// and more generally, get centering_transform
	Eigen::Affine3f rotate_axes_to_cloud;
	rotate_axes_to_cloud = solver.eigenvectors();
	if (rotate_axes_to_cloud.linear().determinant() < 0) rotate_axes_to_cloud.scale(-1); // get a proper rotation
	result_rotate_cloud_to_axes = rotate_axes_to_cloud.inverse();
}

void ModelPatch::getRequiredTSDFValues(Eigen::Matrix3Xf const& cloud_in_tsdf, 
	Eigen::Array3i & result_cell_counts,
	Eigen::Vector3f & result_center_offset)
{
	// get bounding box
	Eigen::Array3f bb_min = cloud_in_tsdf.rowwise().minCoeff();
	Eigen::Array3f bb_max = cloud_in_tsdf.rowwise().maxCoeff();

	// want to contain any fractional cells...I think this is right up to precision
	Eigen::Array3f bb_min_cells_f = (bb_min - params_.patch.border_create) / params_.volume.cell_size - 1;
	Eigen::Array3f bb_max_cells_f = (bb_max + params_.patch.border_create) / params_.volume.cell_size + 1;
	Eigen::Array3i bb_min_cells( (int)bb_min_cells_f[0], (int)bb_min_cells_f[1], (int)bb_min_cells_f[2] ); 
	Eigen::Array3i bb_max_cells( (int)bb_max_cells_f[0], (int)bb_max_cells_f[1], (int)bb_max_cells_f[2] ); 
	result_cell_counts = bb_max_cells - bb_min_cells;
	result_center_offset = -bb_min_cells.cast<float>() * params_.volume.cell_size;
}

void ModelPatch::getCenteringPose(Eigen::Matrix3Xf const& cloud_in_world, 
	Eigen::Vector3f & result_cloud_mean, 
	Eigen::Affine3f & result_rotate_cloud_to_axes,
	Eigen::Array3i & result_cell_counts,
	Eigen::Vector3f & result_center_offset)
{
	getCovarianceRotation(cloud_in_world, result_cloud_mean, result_rotate_cloud_to_axes);

	// center and rotate cloud
	Eigen::Affine3f move_cloud_to_center;
	move_cloud_to_center = Eigen::Translation3f(-result_cloud_mean);
	Eigen::Matrix3Xf cloud_centered_and_rotated = result_rotate_cloud_to_axes * move_cloud_to_center * cloud_in_world;

	// get tsdf values
	getRequiredTSDFValues(cloud_centered_and_rotated, result_cell_counts, result_center_offset);
}

void ModelPatch::getSegmentation(const Frame & frame,
	const Eigen::Affine3f & pose,
	cv::Mat & result_render_segments, 
	cv::Mat & result_consistent_segments,
	cv::Mat & result_segments,
	std::map<int, int> & segment_sizes_map,
	std::map<int, Eigen::Vector3f> segment_normals_map)
{
	boost::timer t;

	/////////////////////
	// first need to render (again!)
	// assume that correct grids are already activated!
    cv::Rect rect(0,0,params_.camera.size.x(), params_.camera.size.y());
	const int rows = rect.height;
	const int cols = rect.width;
	render_buffers_.resetAllBuffers();
    renderModel(params_.camera, pose, render_buffers_);

	result_render_segments = render_buffers_.getImageBufferMask().getMat();

	///////////////////////////
	// now get consistent segments
	// based on getConsistentRenderSegments
	// needs rendered points (really just depths) and normals
	// note how easily this could be put in OpenCL!

	result_consistent_segments = cv::Mat(rect.size(), CV_32SC1, cv::Scalar(0));

	// try points and normals in opencv as well?
	// render:
	cv::Mat mat_render_points = render_buffers_.getImageBufferPoints().getMat();
	cv::Mat mat_render_normals = render_buffers_.getImageBufferNormals().getMat();

	// frame (assumes computed):
	cv::Mat mat_frame_points = frame.image_buffer_points.getMat();
	cv::Mat mat_frame_normals = frame.image_buffer_normals.getMat();

	const float min_dot_product = cos(params_.patch.segments_max_angle * M_PI / 180);

	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			bool is_consistent = true;
			// must have been rendered
			is_consistent = is_consistent && result_render_segments.at<int>(row,col) != 0;
			// must also have a valid normal in frame
			const cv::Vec4f & frame_normal = mat_frame_normals.at<cv::Vec4f>(row,col);
			is_consistent = is_consistent && (frame_normal[0] == frame_normal[0]);

			// depth difference
			float render_z = mat_render_points.at<cv::Vec4f>(row,col)[2];
			float frame_z = mat_frame_points.at<cv::Vec4f>(row,col)[2]; // you also have frame mat_depth, fool...
			is_consistent = is_consistent && fabs(render_z - frame_z) <= params_.patch.segments_max_depth_sigmas * Noise::simpleAxial(frame_z);

			// normal diff
			const cv::Vec4f & render_normal = mat_render_normals.at<cv::Vec4f>(row,col);

			// yuck..
			Eigen::Vector3f n1(render_normal[0], render_normal[1], render_normal[2]);
			Eigen::Vector3f n2(frame_normal[0], frame_normal[1], frame_normal[2]);
			is_consistent = is_consistent && n1.dot(n2) >= min_dot_product;

			if (is_consistent) {
				result_consistent_segments.at<int>(row,col) = result_render_segments.at<int>(row,col);
			}
		}
	}

	///////////////////////////
	// finally, merge into and create segments
	// based on segmentByMerging

	std::map<int, int> input_to_disjoint_set_map;
	std::map<int, int> disjoint_set_to_output_map;

	DisjointSet disjoint_set(rows * cols);
	typedef std::map<int, Eigen::Vector3f> NormalMapT;
	NormalMapT disjoint_set_normal_map;
	std::vector<MergeEdge> edge_v;

	// right, down-left, down, down-right
	const static int deltas[] = {0,1, 1,-1, 1,0, 1,1};

	// create edges
	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			// skip invalid points
			const cv::Vec4f & frame_normal = mat_frame_normals.at<cv::Vec4f>(row,col);
			if (frame_normal[0] != frame_normal[0]) continue;

			const int index = row * cols + col;

			cv::Vec4f point = mat_frame_points.at<cv::Vec4f>(row,col);
			Eigen::Vector3f point_eigen (point[0], point[1], point[2]);
			cv::Vec4f normal = mat_frame_normals.at<cv::Vec4f>(row,col);
			Eigen::Vector3f normal_eigen (normal[0], normal[1], normal[2]);

			// get the input component
			int input_component = result_consistent_segments.at<int>(row,col);
			if (input_component > 0) {
				// if we have an input component, need to map to disjoint set
				std::map<int,int>::iterator find_iter = input_to_disjoint_set_map.find(input_component);
				if (find_iter == input_to_disjoint_set_map.end()) {
					// This is the first time we've seen this input component
					// Map the input component to this disjoint set
					input_to_disjoint_set_map[input_component] = index;
					disjoint_set_to_output_map[index] = input_component;
					disjoint_set_normal_map[index] = normal_eigen;
				}
				else {
					// Merge this point into the component
					int new_disjoint_set = disjoint_set.connect(index, find_iter->second);
					// add the normal
					// Note that we now have "old" disjoint sets in component_mean_normal_map and disjoint_set_to_output_map
					input_to_disjoint_set_map[input_component] = new_disjoint_set;
					disjoint_set_to_output_map[new_disjoint_set] = input_component;
					disjoint_set_normal_map[new_disjoint_set] = disjoint_set_normal_map[find_iter->second] + normal_eigen;
				}
			}
			else {
				// no input component
				disjoint_set_normal_map[index] = normal_eigen;
			}			

			// go through "delta" neighbors
			for (int n = 0; n < 4; ++n) {
				int n_row = row + deltas[2*n];
				int n_col = col + deltas[2*n+1];

				// skip invalid neighbors
				if (n_row < 0 || n_row >= rows || n_col < 0 || n_col >= cols) continue;

				const int n_index = n_row * cols + n_col;

				cv::Vec4f n_point = mat_frame_points.at<cv::Vec4f>(n_row,n_col);
				Eigen::Vector3f n_point_eigen (n_point[0], n_point[1], n_point[2]);
				cv::Vec4f n_normal = mat_frame_normals.at<cv::Vec4f>(n_row,n_col);
				Eigen::Vector3f n_normal_eigen (n_normal[0], n_normal[1], n_normal[2]);

				if (n_normal[0] != n_normal[0]) continue;

				// don't put an edge if already the same component (though it is safe to do so)
				if (input_component > 0 && result_consistent_segments.at<int>(n_row,n_col) == input_component) continue;

				// check distance
				if (abs(point_eigen.z() - n_point_eigen.z()) > params_.patch.segments_max_depth_sigmas * Noise::simpleAxial(point_eigen.z())) continue;

				// use normal dot as weight?
				float dot_product = normal_eigen.dot(n_normal_eigen);

				edge_v.push_back(MergeEdge());
				MergeEdge& edge = edge_v.back();
				edge.a = index;
				edge.b = n_index;
				edge.w = 1-dot_product; // use 1-dot so sort ascending works
			}
		}
	}
	std::sort(edge_v.begin(), edge_v.end());

	/////////////////////////
	// need to normalize component_mean_normal_map
	for (NormalMapT::iterator iter = disjoint_set_normal_map.begin(); iter != disjoint_set_normal_map.end(); ++iter) {
		iter->second.normalize();
	}


	///////////////////////////////
	// merging
	// for each edge, if current normal dot product is ok, merge
	// edges will only be between valid points
	// can assume the normal map is complete
	for (std::vector<MergeEdge>::iterator iter = edge_v.begin(); iter != edge_v.end(); ++iter) {
		int a_set = disjoint_set.find(iter->a);
		int b_set = disjoint_set.find(iter->b);
		if (a_set == b_set) continue;
		// different sets, so check normal agreement:
		NormalMapT::iterator a_normal_find = disjoint_set_normal_map.find(a_set);
		NormalMapT::iterator b_normal_find = disjoint_set_normal_map.find(b_set);
		const Eigen::Vector3f& a_normal = a_normal_find->second;
		const Eigen::Vector3f& b_normal = b_normal_find->second;
		float dot_product = a_normal.dot(b_normal);
		if (dot_product < min_dot_product) continue;

		// needed to check and maintain input component mapping to output
		int a_output_component = disjoint_set_to_output_map[a_set];
		int b_output_component = disjoint_set_to_output_map[b_set];
		// if a and b are part of different input sets, don't merge
		if (a_output_component != 0 && b_output_component != 0) continue;

		// join:
		int a_size = disjoint_set.size(a_set);
		int b_size = disjoint_set.size(b_set);
		int new_set = disjoint_set.connect(a_set, b_set);
		disjoint_set_normal_map[new_set] = (a_size * a_normal + b_size * b_normal).normalized();

		// also update disjoint_set_to_output_map
		int new_set_output_component = disjoint_set_to_output_map[new_set];
		// if the new_set output component is zero, make sure we didn't lose our input component
		if (new_set_output_component == 0) {
			if (a_output_component != 0) disjoint_set_to_output_map[new_set] = a_output_component;
			else if (b_output_component != 0) disjoint_set_to_output_map[new_set] = b_output_component;
			// else we don't have an input component for either set
		}
	}

	////////////////////
	// merge in small segments at the end?
	for (std::vector<MergeEdge>::iterator iter = edge_v.begin(); iter != edge_v.end(); ++iter) {
		int a_set = disjoint_set.find(iter->a);
		int b_set = disjoint_set.find(iter->b);
		if (a_set == b_set) continue;
		// different sets, make sure both are large enough
		int a_size = disjoint_set.size(a_set);
		int b_size = disjoint_set.size(b_set);
		if (a_size >= params_.patch.segments_min_size && b_size >= params_.patch.segments_min_size) continue;

		// needed to check and maintain input component mapping to output
		int a_output_component = disjoint_set_to_output_map[a_set];
		int b_output_component = disjoint_set_to_output_map[b_set];
		// if a and b are part of different input sets, don't merge
		if (a_output_component != 0 && b_output_component != 0) continue;

		int new_set = disjoint_set.connect(a_set, b_set);
		NormalMapT::iterator a_normal_find = disjoint_set_normal_map.find(a_set);
		NormalMapT::iterator b_normal_find = disjoint_set_normal_map.find(b_set);
		const Eigen::Vector3f& a_normal = a_normal_find->second;
		const Eigen::Vector3f& b_normal = b_normal_find->second;
		disjoint_set_normal_map[new_set] = (a_size * a_normal + b_size * b_normal).normalized();

		// also update disjoint_set_to_output_map
		int new_set_output_component = disjoint_set_to_output_map[new_set];
		// if the new_set output component is zero, make sure we didn't lose our input component
		if (new_set_output_component == 0) {
			if (a_output_component != 0) disjoint_set_to_output_map[new_set] = a_output_component;
			else if (b_output_component != 0) disjoint_set_to_output_map[new_set] = b_output_component;
			// else we don't have an input component for either set
		}
	}


	//////////////////////
	// form output

	result_segments = cv::Mat(rows, cols, CV_32SC1, cv::Scalar(0));

	// this is so that every output component gets a fresh segment_id
	// will you actually use this later?? Probably doesn't matter
	// just needs to be unique relative to rendered volumes segment_id's
	// this ternary op is a hack around this happening on the first frame....
	// if you want colors to be consistent, you may want this later
	// god this is stupid....
	int new_segment_id_counter = grid_list_.size() + 1; // segment_id + 1

	// key issue: may have new components which don't correspond to existing segment_id's
	// in other words, they are not in disjoint_set_to_output_map
	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			// skip invalid points
			// sure do this a lot...
			const cv::Vec4f & frame_normal = mat_frame_normals.at<cv::Vec4f>(row,col);
			if (frame_normal[0] != frame_normal[0]) continue;

			int index = row * cols + col;
			int set = disjoint_set.find(index);

			std::map<int,int>::iterator find_component = disjoint_set_to_output_map.find(set);

			// figure out which component
			int component = -1;
			// not sure why this would ever be .end(), actually...
			// looking above, it appears that it can definitely be 0 if there was nothing rendered to the frame pixel
			if (find_component == disjoint_set_to_output_map.end() || find_component->second == 0) {
				component = new_segment_id_counter++;
				disjoint_set_to_output_map[set] = component;
			}
			else {
				component = find_component->second;
			}

			// ok...now we have the component.  

			// lazy and slow...just overwrite in the map every time
			// EVERY PIXEL (note you would still have to search the map anyway...)
			segment_sizes_map[component] = disjoint_set.size(set);
			segment_normals_map[component] = disjoint_set_normal_map[set];

			result_segments.at<int>(row,col) = component;
		}
	}

	cout << "getSegmentation time: " << t.elapsed() << endl;
}
