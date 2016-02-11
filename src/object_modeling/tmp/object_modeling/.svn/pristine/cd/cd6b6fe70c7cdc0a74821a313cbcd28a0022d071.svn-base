#include "model_moving_volume_grid.h"

#include "frustum.h"

#include "MeshUtilities.h"

#include "util.h"

// for parts of load and save
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>

#include <boost/tuple/tuple_io.hpp>

ModelMovingVolumeGrid::ModelMovingVolumeGrid(boost::shared_ptr<OpenCLAllKernels> all_kernels, VolumeModelerAllParams const& params, boost::shared_ptr<Alignment> alignment_ptr, RenderBuffers const& render_buffers)
	: ModelGrid(all_kernels, params, alignment_ptr, render_buffers),
	prepare_for_render_loop_closure_(false)
{
    resetThisClass();
}

ModelMovingVolumeGrid::ModelMovingVolumeGrid(ModelMovingVolumeGrid const& other)
	: ModelGrid(other),
	prepare_for_render_loop_closure_(other.prepare_for_render_loop_closure_),
	moving_tsdf_ptr_(other.moving_tsdf_ptr_->clone()),
	moving_tsdf_pose_(other.moving_tsdf_pose_),
	keyframe_to_grid_creation_(other.keyframe_to_grid_creation_),
	moving_volume_block_offset_(other.moving_volume_block_offset_),
	block_index_to_keyframe_(other.block_index_to_keyframe_)
{
}

ModelMovingVolumeGrid* ModelMovingVolumeGrid::clone()
{
	return new ModelMovingVolumeGrid(*this);
}

void ModelMovingVolumeGrid::resetMovingVolume()
{
	//const Eigen::Array3i volume_cell_counts(params_.volume.cell_count.x(), params_.volume.cell_count.y(), params_.volume.cell_count.z());
	const int s = params_.moving_volume_grid.blocks_in_moving_volume * params_.grid.grid_size + 1 /* built in overlap? */;
	const Eigen::Array3i volume_cell_counts(s,s,s);

    moving_tsdf_ptr_.reset(new OpenCLTSDF(all_kernels_,
        params_.camera.focal[0], params_.camera.focal[1],
        params_.camera.center[0], params_.camera.center[1],
        params_.camera.size[0], params_.camera.size[1],
        volume_cell_counts[0], volume_cell_counts[1], volume_cell_counts[2],
        params_.volume.cell_size,
        params_.volume.max_weight_icp,
        params_.volume.max_weight_color,
        params_.volume.use_most_recent_color,
            params_.volume.min_truncation_distance));
}

void ModelMovingVolumeGrid::resetThisClass()
{
	resetMovingVolume();
    moving_tsdf_pose_ = Eigen::Affine3f::Identity();
	keyframe_to_grid_creation_.clear();
	moving_volume_block_offset_ = Eigen::Array3i::Zero();
	block_index_to_keyframe_.clear();
}

void ModelMovingVolumeGrid::reset()
{
    ModelGrid::reset();
	resetThisClass();
}

void ModelMovingVolumeGrid::prepareForRenderCurrent()
{
	// todo: the right thing here...probably
	// in particular, activating for non-loop closure is irrelevant
    ModelGrid::prepareForRenderCurrent();
    prepare_for_render_loop_closure_ = false;
}

void ModelMovingVolumeGrid::renderModel(
        const ParamsCamera & params_camera,
        const Eigen::Affine3f & model_pose,
        RenderBuffers & render_buffers)
{
    // related idea: move more of the graph structure and so forth INTO the stupid model as opposed to in the volume modeler
	if (prepare_for_render_loop_closure_) {
		// just do what base class does??  assumes correct activation
        ModelGrid::renderModel(params_camera, model_pose, render_buffers);
	}
	else {
        render_buffers.setSize(params_camera.size.x(), params_camera.size.y());
		render_buffers.resetAllBuffers();

        const int mask_value = 1;
        moving_tsdf_ptr_->renderFrame(model_pose * moving_tsdf_pose_,
            params_camera.focal, params_camera.center, params_camera.min_max_depth,
            false, mask_value,
			render_buffers);
	}
}

void ModelMovingVolumeGrid::updateModel(
	Frame & frame,
	const Eigen::Affine3f & model_pose)
{
	// note, not ModelGrid!!
	ModelBase::updateModel(frame, model_pose);

	Eigen::Vector3f tsdf_center_world = moving_tsdf_pose_ * moving_tsdf_ptr_->getSphereCenter();
	Eigen::Vector3f camera_center_world = model_pose.inverse() * Eigen::Vector3f(0,0,params_.moving_volume_grid.camera_center_distance);

	const float block_size = params_.volume.cell_size * params_.grid.grid_size;
	Eigen::Array3i block_shifts = EigenUtilities::truncateVector3fToInt( (camera_center_world - tsdf_center_world) / block_size );

	// needed?
	//grids_updated_last_call_.clear();

	// todo: these in one structure?
	// we need to:
	// - clear ALL blocks into graph edges or nothing
	// - 
	std::vector<int> grids_added;

	// need block to grid mapping
	typedef std::map<BlockIndexT, int> MapBlockInt;
	MapBlockInt block_to_grid_added;

	// now have to apply block_shifts
	if ( (block_shifts.abs() > 0).any() ) {
		OpenCLTSDFPtr old_volume = moving_tsdf_ptr_;
		Eigen::Affine3f old_tsdf_pose = moving_tsdf_pose_;
		Eigen::Affine3f shift_translation;
		shift_translation = Eigen::Translation3f( (block_shifts.cast<float>() * block_size) );
		Eigen::Affine3f new_tsdf_pose = shift_translation * old_tsdf_pose;

		// also the (redundant) block offset
		Eigen::Array3i old_moving_volume_block_offset = moving_volume_block_offset_;
		Eigen::Array3i new_moving_volume_block_offset = old_moving_volume_block_offset + block_shifts;

		///////////////
		// extract spit-out grids
		// this appears to work:
		{
			Eigen::Array3i min_block(0,0,0);
			Eigen::Array3i max_block(0,0,0);
			for (int i = 0; i < 3; ++i) {
				min_block[i] = block_shifts[i] >= 0 ? 0 : params_.moving_volume_grid.blocks_in_moving_volume + block_shifts[i];
				max_block[i] = block_shifts[i] >= 0 ? block_shifts[i] : params_.moving_volume_grid.blocks_in_moving_volume;
			}

			for (int block_x = min_block.x(); block_x < max_block.x(); ++block_x) {
				const int voxel_x = block_x * params_.grid.grid_size;
				for (int block_y = 0; block_y < params_.moving_volume_grid.blocks_in_moving_volume; ++block_y) {
					const int voxel_y = block_y * params_.grid.grid_size;
					for (int block_z = 0; block_z < params_.moving_volume_grid.blocks_in_moving_volume; ++block_z) {
						const int voxel_z = block_z * params_.grid.grid_size;
						Eigen::Array3i voxel = Eigen::Array3i(voxel_x, voxel_y, voxel_z);
						int added = appendGridCellFromMovingVolume(*old_volume, old_tsdf_pose, voxel);
						Eigen::Array3i block = Eigen::Array3i(block_x, block_y, block_z);
						BlockIndexT block_world_index = EigenUtilities::array3iToTuple(block + old_moving_volume_block_offset);
						block_to_grid_added[block_world_index] = added; // -1 if none added
						if (added >= 0) {
							grids_added.push_back(added);
						}
					}
				}
			}

			for (int block_y = min_block.y(); block_y < max_block.y(); ++block_y) {
				const int voxel_y = block_y * params_.grid.grid_size;
				for (int block_x = 0; block_x < params_.moving_volume_grid.blocks_in_moving_volume; ++block_x) {
					if (block_x >= min_block.x() && block_x < max_block.x()) continue; // skip blocks we already copied
					const int voxel_x = block_x * params_.grid.grid_size;
					for (int block_z = 0; block_z < params_.moving_volume_grid.blocks_in_moving_volume; ++block_z) {
						const int voxel_z = block_z * params_.grid.grid_size;
						Eigen::Array3i voxel = Eigen::Array3i(voxel_x, voxel_y, voxel_z);
						int added = appendGridCellFromMovingVolume(*old_volume, old_tsdf_pose, voxel);
						Eigen::Array3i block = Eigen::Array3i(block_x, block_y, block_z);
						BlockIndexT block_world_index = EigenUtilities::array3iToTuple(block + old_moving_volume_block_offset);
						block_to_grid_added[block_world_index] = added; // -1 if none added
						if (added >= 0) {
							grids_added.push_back(added);
						}
					}
				}
			}

			for (int block_z = min_block.z(); block_z < max_block.z(); ++block_z) {
				const int voxel_z = block_z * params_.grid.grid_size;
				for (int block_x = 0; block_x < params_.moving_volume_grid.blocks_in_moving_volume; ++block_x) {
					if (block_x >= min_block.x() && block_x < max_block.x()) continue; // skip blocks we already copied
					const int voxel_x = block_x * params_.grid.grid_size;

					for (int block_y = 0; block_y < params_.moving_volume_grid.blocks_in_moving_volume; ++block_y) {
						if (block_y >= min_block.y() && block_y < max_block.y()) continue; // skip blocks we already copied
						const int voxel_y = block_y * params_.grid.grid_size;
						Eigen::Array3i voxel = Eigen::Array3i(voxel_x, voxel_y, voxel_z);
						int added = appendGridCellFromMovingVolume(*old_volume, old_tsdf_pose, voxel);
						Eigen::Array3i block = Eigen::Array3i(block_x, block_y, block_z);
						BlockIndexT block_world_index = EigenUtilities::array3iToTuple(block + old_moving_volume_block_offset);
						block_to_grid_added[block_world_index] = added; // -1 if none added
						if (added >= 0) {
							grids_added.push_back(added);
						}
					}
				}
			}
		}


		///////////////
		// move the volume
		{
			boost::timer t_moving_copy;
			// this is a full copy...obviously not ideal
			// need special axis-aligned functions (and ultimately circular buffer)

			boost::timer t_reset;
			resetMovingVolume();
			cout << "[TIMING] resetMovingVolume (subset of t_moving_copy): " << t_reset.elapsed() << endl;

			// axis align (still slow) way:
			Eigen::Array3i voxel_shifts = block_shifts * params_.grid.grid_size;

			// can now handle negative origin values
			boost::timer t_copy;
			moving_tsdf_ptr_->copyVolumeAxisAligned(*old_volume, voxel_shifts, Eigen::Array3i(0,0,0));
			cout << "[TIMING] copyVolumeAxisAligned (subset of t_moving_copy): " << t_copy.elapsed() << endl;

			moving_tsdf_pose_ = new_tsdf_pose;
			cout << "[TIMING] t_moving_copy: " << t_moving_copy.elapsed() << endl;

			// also update blocks at the same time as pose?
			moving_volume_block_offset_ = new_moving_volume_block_offset;
		}

		///////////
		// todo:
		// copy "active" blocks into new volume
		if (params_.moving_volume_grid.debug_disable_merge_on_shift) {
			cout << "WARNING: debug_disable_merge_on_shift" << endl;
		}
		else {
			mergeVolumesIntoMovingVolume();
		}

	} // end of shifting


    // can leave buffer_segments empty if which_segment = 0;
    // but could also use to for masking...
    ImageBuffer buffer_segments(all_kernels_->getCL());
    moving_tsdf_ptr_->addFrame(model_pose * moving_tsdf_pose_, frame.image_buffer_depth, frame.image_buffer_color, frame.image_buffer_segments, frame.image_buffer_add_depth_weights, frame.image_buffer_add_color_weights, 0);

	//// may not have current_keyframe_ here
	int previous_keyframe = current_keyframe_; // this is ugly
	bool changed_keyframe = updateKeyframe(frame);
	//// will always have current_keyframe_ here

	///////////////
	// update block keyframe association for all blocks in (new) current keyframe
	// also link to previous keyframe if valid and different
	{
		const Eigen::Array3i volume_cell_counts = Eigen::Array3i::Zero() + params_.grid.grid_size;

		Eigen::Affine3f camera_pose = model_pose.inverse(); // earlier??

		// can do this once (frustum moves)
		Eigen::Array3f min_point, max_point;
        getAABB(volume_cell_counts, params_.volume.cell_size, min_point, max_point);

		Eigen::Array3i min_block = Eigen::Array3i::Zero();
		Eigen::Array3i max_block = min_block + params_.moving_volume_grid.blocks_in_moving_volume;
		for (int block_x = min_block.x(); block_x < max_block.x(); ++block_x) {
			const int voxel_x = block_x * params_.grid.grid_size;
			for (int block_y = 0; block_y < params_.moving_volume_grid.blocks_in_moving_volume; ++block_y) {
				const int voxel_y = block_y * params_.grid.grid_size;
				for (int block_z = 0; block_z < params_.moving_volume_grid.blocks_in_moving_volume; ++block_z) {
					const int voxel_z = block_z * params_.grid.grid_size;
					Eigen::Array3i voxel(voxel_x, voxel_y, voxel_z);
					Eigen::Array3i block_array(block_x, block_y, block_z);
					Eigen::Array3i block_array_world = block_array + moving_volume_block_offset_;
					BlockIndexT block_index_world = EigenUtilities::array3iToTuple(block_array_world);

					Eigen::Affine3f block_pose = getBlockTransform(moving_tsdf_pose_, voxel);
					Frustum frustum_in_tsdf_space(params_.camera, block_pose.inverse() * camera_pose);

					if (frustum_in_tsdf_space.doesAABBIntersect(min_point, max_point)) {
						block_index_to_keyframe_[block_index_world].insert(current_keyframe_);
						if (previous_keyframe >= 0 && previous_keyframe != current_keyframe_) {
                            block_index_to_keyframe_[block_index_world].insert(previous_keyframe);
						}
					}
				}
			}
		}
	}

	// update record of keyframe and block creation (useful?)
	std::copy(grids_added.begin(), grids_added.end(), std::back_inserter(keyframe_to_grid_creation_[current_keyframe_]));

	// ok...so now we go through all blocks that were shifted out
	// clear the block_index_to_keyframe regardless
	// and link with those keyframes for valid blocks
	BOOST_FOREACH(MapBlockInt::value_type & p, block_to_grid_added) {
		std::set<int> & keyframes_for_block = block_index_to_keyframe_[p.first];

#if 0
		// debug remove
		if (p.second >= 0) {
			cout << "--------- pair: " << p.first << " -> " << p.second << endl;
			cout << "keyframes: ";
			BOOST_FOREACH(int keyframe, keyframes_for_block) {
				cout << keyframe << " ";
			}
			cout << endl;
		}
#endif


		if (p.second >= 0) {
			std::vector<int> ugly_block_list(1, p.second);
			BOOST_FOREACH(int const& keyframe, keyframes_for_block) {
				addEdgesToKeyframeGraphForVolumes(keyframe, ugly_block_list);
			}
		}
		keyframes_for_block.clear(); // dump all keyframes for block (having now added the appropriate edges)
	}
}

Eigen::Affine3f ModelMovingVolumeGrid::getBlockTransform(Eigen::Affine3f const& tsdf_pose, Eigen::Array3i const& voxel)
{
	Eigen::Vector3f block_to_moving_volume = voxel.matrix().cast<float>() * params_.volume.cell_size;
	Eigen::Affine3f block_transform = tsdf_pose * Eigen::Translation3f(block_to_moving_volume);
	return block_transform;
}

int ModelMovingVolumeGrid::appendGridCellFromMovingVolume(OpenCLTSDF & tsdf, Eigen::Affine3f const& tsdf_pose, Eigen::Array3i const& voxel)
{
	// try using border...
	const int s = params_.grid.grid_size + 1 /* built in border */;
	Eigen::Array3i box_size(s,s,s);

	//boost::timer t_test;
	bool box_contains_surface = tsdf.doesBoxContainSurface(voxel, box_size);
	//cout << "Time for t_test: " << t_test.elapsed() << endl;

	if (!box_contains_surface) return -1;

	Eigen::Affine3f block_transform = getBlockTransform(tsdf_pose, voxel);

	//boost::timer t_append_copy;
	appendNewGridStruct(block_transform, Eigen::Affine3f::Identity(), box_size);
	grid_list_.back()->tsdf_ptr->copyVolumeAxisAligned(tsdf, voxel, Eigen::Array3i(0,0,0));
	//cout << "Time for append and copy: " << t_append_copy.elapsed() << endl;

	return grid_list_.size() - 1;
}


void ModelMovingVolumeGrid::save(fs::path const& folder)
{
	ModelGrid::save(folder);
	
	moving_tsdf_ptr_->save(folder);

	{
		fs::path filename = folder / "model_moving_volume_grid_tsdf_pose.txt";
		std::ofstream file (filename.string().c_str());
		file << EigenUtilities::transformToString(moving_tsdf_pose_) << endl;
	}

	{
		fs::path filename = folder / "model_moving_volume_archive.txt";
		std::fstream file(filename.string().c_str(), std::ios::out);
		boost::archive::text_oarchive a(file);

		a & keyframe_to_grid_creation_;
		a & block_index_to_keyframe_;
		Eigen::Vector3i block_offset_temp = moving_volume_block_offset_.matrix();
		a & block_offset_temp;
	}
}

void ModelMovingVolumeGrid::load(fs::path const& folder)
{
	ModelGrid::load(folder);

	moving_tsdf_ptr_->load(folder);
	
	{
		fs::path filename = folder / "model_moving_volume_grid_tsdf_pose.txt";
		std::ifstream file(filename.string().c_str());
		std::string line;
		std::getline(file, line);
		moving_tsdf_pose_ = EigenUtilities::stringToTransform(line);
	}

	{
		fs::path filename = folder / "model_moving_volume_archive.txt";
		std::fstream file(filename.string().c_str(), std::ios::in);
		boost::archive::text_iarchive a(file);

		a & keyframe_to_grid_creation_;
		a & block_index_to_keyframe_;
		Eigen::Vector3i block_offset_temp;
		a & block_offset_temp;
		moving_volume_block_offset_ = block_offset_temp.array();
	}
}


void ModelMovingVolumeGrid::refreshUpdateInterface()
{
	ModelGrid::refreshUpdateInterface();

	if (update_interface_) {
		const Eigen::Vector4ub color(255,0,255,0);

		MeshVertexVectorPtr bounding_lines_ptr (new MeshVertexVector);
		moving_tsdf_ptr_->getBoundingLines(moving_tsdf_pose_, color, *bounding_lines_ptr);
		update_interface_->updateLines("moving_volume", bounding_lines_ptr);
	}
}

void ModelMovingVolumeGrid::createG2OPoseGraphKeyframes(
		G2OPoseGraph & pose_graph,
		std::map<int, int> & keyframe_to_vertex_map,
		std::map<int, int> & volume_to_vertex_map)
{
	volume_to_vertex_map.clear();
	keyframe_to_vertex_map.clear();

	// TODO: call the base class version of this first...as we are simply adding additional edges and vertices?
	for (int i = 0; i < (int)keyframe_list_.size(); ++i) {
		KeyframeStruct & keyframe = *keyframe_list_[i];
		CameraStruct & keyframe_cam = *camera_list_[keyframe.camera_index];
		keyframe_to_vertex_map[i] = pose_graph.addVertex(EigenUtilities::getIsometry3d(keyframe_cam.pose), false);
	}
	int volume_count = getVolumeCount();
	for (int i = 0; i < volume_count; ++i) {
		volume_to_vertex_map[i] = pose_graph.addVertex(EigenUtilities::getIsometry3d(getVolumePoseExternal(i)), false);
	}

	// insert the normal edges
	BOOST_FOREACH(EdgeStruct const& e, keyframe_edge_list_) {
		addEdgeToPoseGraph(pose_graph, keyframe_to_vertex_map, volume_to_vertex_map, e);
	}

	// inject extra edges
	// do I want the center of the volume?  Probably...
	Eigen::Translation3f center_offset(moving_tsdf_ptr_->getSphereCenter());
	Eigen::Affine3f moving_volume_pose_for_graph = moving_tsdf_pose_ * center_offset;
	int moving_volume_vertex = pose_graph.addVertex(EigenUtilities::getIsometry3d(moving_volume_pose_for_graph), true); // fixed?

	typedef std::map<int,int> Map;
	Map keyframes_within_distance;
	getVerticesWithinDistanceSimpleGraph(keyframe_to_keyframe_edges_, current_keyframe_, params_.loop_closure.keyframe_graph_distance, keyframes_within_distance);

	// check against moving volume?
	Map keyframes_within_distance_seeing_volume;
	{
		Eigen::Array3f min_point, max_point;
		moving_tsdf_ptr_->getAABB(min_point, max_point);
		BOOST_FOREACH(Map::value_type & p, keyframes_within_distance) {
			Eigen::Affine3f camera_pose = camera_list_[keyframe_list_[p.first]->camera_index]->pose;
			Frustum frustum_in_tsdf_space(params_.camera, moving_tsdf_pose_.inverse() * camera_pose);
			if (frustum_in_tsdf_space.doesAABBIntersect(min_point, max_point)) {
				keyframes_within_distance_seeing_volume.insert(p);
			}
		}
	}

	// add the bonus edges
	BOOST_FOREACH(Map::value_type & p, keyframes_within_distance_seeing_volume) {
		KeyframeStruct const& keyframe = *keyframe_list_[p.first];
		Eigen::Affine3f const& keyframe_pose = camera_list_[keyframe.camera_index]->pose;
		Eigen::Affine3f keyframe_pose_inverse = keyframe_pose.inverse();
		Eigen::Affine3f relative_pose = keyframe_pose_inverse * moving_volume_pose_for_graph;
		bool added = pose_graph.addEdge(keyframe_to_vertex_map[p.first], moving_volume_vertex, EigenUtilities::getIsometry3d(relative_pose));

		// also fix all of these keyframes??
		if (params_.moving_volume_grid.debug_fix_g2o_vertices_for_keyframes) {
			pose_graph.setVertexFixed(keyframe_to_vertex_map[p.first], true);
		}
	}

	// dup from ModelGrid version:
	pose_graph.setVerbose(true); // debug
	int fixed_vertex = keyframe_to_vertex_map.find(current_keyframe_)->second;
	pose_graph.setVertexFixed(fixed_vertex, true);
}

void ModelMovingVolumeGrid::generateMesh(
	MeshVertexVector & vertex_list,
	TriangleVector & triangle_list)
{
	ModelGrid::generateMesh(vertex_list, triangle_list);

	if (!params_.moving_volume_grid.debug_no_moving_mesh) {
		MeshVertexVector moving_vertex_list;
		TriangleVector moving_triangle_list;
		moving_tsdf_ptr_->generateMesh(moving_vertex_list, moving_triangle_list);
		MeshUtilities::transformMeshVertices(moving_tsdf_pose_, moving_vertex_list);
		MeshUtilities::appendMesh(vertex_list, triangle_list, moving_vertex_list, moving_triangle_list);
	}
}

void ModelMovingVolumeGrid::mergeVolumesIntoMovingVolume()
{
	// get "active" volumes which intersect the moving volume
	if (params_.loop_closure.activation_mode != ACTIVATION_MODE_KEYFRAME_GRAPH) {
		cout << "WARNING: skipping mergeVolumesIntoMovingVolume because params_.loop_closure.activation_mode != ACTIVATION_MODE_KEYFRAME_GRAPH" << endl;
		return;
	}

	if (params_.loop_closure.debug_disable_merge) {
		cout << "WARNING: debug_disable_merge" << endl;
		return ;
	}

	if (params_.loop_closure.debug_merge_show_points) {
		// refresh here for consistency
		refreshUpdateInterface();

		// moving points
#if 0
		MeshVertexVectorPtr point_vertices(new MeshVertexVector);
		moving_tsdf_ptr_->getPrettyVoxelCenters(moving_tsdf_pose_, debug_show_empty, *point_vertices);
		update_interface_->updatePointCloud("debug_merge_show_points", point_vertices);
		cout << "debug_merge_show_points pause..." << endl;
		cv::waitKey();
#endif
	}

	if (params_.moving_volume_grid.debug_clipping) {
		refreshUpdateInterface(); // consistency
	}

    activateKeyframeGraph(current_keyframe_, false);
	for (int i = 0; i < grid_list_.size(); ++i) {
		GridStruct & grid_struct = *grid_list_[i];
		if (!grid_struct.active) continue;
		
		Eigen::Affine3f relative_pose = grid_struct.getExternalTSDFPose().inverse() * moving_tsdf_pose_;

		// should also skip those grids not falling inside volume at all...
		// inverse??
        if (!moving_tsdf_ptr_->couldOtherVolumeIntersect(*grid_struct.tsdf_ptr, relative_pose)) continue;


		// debug
		if (params_.loop_closure.debug_merge_show_points) {
			Eigen::Affine3f grid_struct_pose = grid_struct.getExternalTSDFPose();

			// grid struct points
			{
				MeshVertexVectorPtr point_vertices(new MeshVertexVector);
				grid_struct.tsdf_ptr->getPrettyVoxelCenters(grid_struct_pose, *point_vertices);
				update_interface_->updatePointCloud("debug_merge_show_points", point_vertices);
			}

			// also mesh?
			{
				MeshPtr mesh_ptr (new Mesh);

				grid_struct.tsdf_ptr->generateMesh(mesh_ptr->vertices, mesh_ptr->triangles);
				MeshUtilities::transformMeshVertices(grid_struct_pose, mesh_ptr->vertices);
				update_interface_->updateMesh("debug_merge_show_points_mesh", mesh_ptr);
			}

			// also dump values??
			// tells me nothing useful
			if (false) {
				typedef std::pair<Eigen::Vector3f, std::pair<float,float> > PANDP;
				std::vector<PANDP> v;
				grid_struct.tsdf_ptr->getVoxelDepthAndWeightValues(grid_struct_pose, v);
				BOOST_FOREACH(PANDP const& p, v) {
					cout << p.second.first << " " << p.second.second << endl;
				}
			}

			cout << "debug_merge_show_points pause..." << endl;
			cv::waitKey();
		}

		//////////////////////////
		///// MODIFY VOLUMES
		moving_tsdf_ptr_->addVolume(*grid_struct.tsdf_ptr, relative_pose);
		{
			// clip volumes by moving volume
			Eigen::Affine3f adjust_clipping_pose;
			// instead of a full cell border, you might just do epsilon?
			Eigen::Array3i volume_cell_counts_for_scaling = moving_tsdf_ptr_->getVolumeCellCounts() - 1 - 2; // this 2 is one cell border all over
			const float cell_size = moving_tsdf_ptr_->getVolumeCellSize();
			adjust_clipping_pose = Eigen::Scaling(volume_cell_counts_for_scaling.matrix().cast<float>() * cell_size);
			adjust_clipping_pose.pretranslate(Eigen::Vector3f(cell_size, cell_size, cell_size)); // one cell border all over...
			Eigen::Affine3f set_value_in_box_pose = relative_pose * adjust_clipping_pose;
			const static Eigen::Array4ub null_color (0,0,0,0);

			if (params_.moving_volume_grid.debug_clipping) {
				{
					// clipping box
					// see getVolumeCorners for why (2,2,2)...
					//std::vector<Eigen::Vector3f> corners = OpenCLTSDF::getVolumeCorners(Eigen::Array3i(2,2,2), 1, grid_struct.getExternalTSDFPose() * set_value_in_box_pose);
					std::vector<Eigen::Vector3f> corners = getBoxCorners(Eigen::Array3f(0,0,0), Eigen::Array3f(1,1,1), grid_struct.getExternalTSDFPose() * set_value_in_box_pose);
					Eigen::Vector4ub color_eigen(0,255,255,255);
					MeshVertexVectorPtr vertices_ptr (new MeshVertexVector);
					getLinesForBoxCorners(corners, color_eigen, *vertices_ptr);
					update_interface_->updateLines("debug_clipping_moving_lines", vertices_ptr);
				}

				{
					MeshVertexVectorPtr vertices_ptr (new MeshVertexVector);
					// grid shell
					//Eigen::Vector4ub color_eigen = getColorEigen(i+1);
					MeshPtr mesh_ptr (new Mesh);
					Eigen::Vector4ub color_eigen(255,255,255,255);
					grid_struct.tsdf_ptr->getBoundingLines(grid_struct.getExternalTSDFPose(), color_eigen, *vertices_ptr);
					update_interface_->updateLines("debug_clipping_grid_lines", vertices_ptr);
				}

				{
					// grid mesh
					MeshPtr mesh_ptr (new Mesh);
					grid_struct.tsdf_ptr->generateMesh(mesh_ptr->vertices, mesh_ptr->triangles);
					MeshUtilities::transformMeshVertices(grid_struct.getExternalTSDFPose(), mesh_ptr->vertices);
					update_interface_->updateMesh("debug_clipping_grid_mesh", mesh_ptr);
				}
				cout << "debug_clipping pause..." << endl;
				cv::waitKey();
			}

			///////////////////////////
			// actually do it!
			grid_struct.tsdf_ptr->setValueInBox(grid_struct.tsdf_ptr->getEmptyDValue(), 0, null_color, 0, set_value_in_box_pose);


			if (params_.moving_volume_grid.debug_clipping) {
				{
					// grid mesh
					MeshPtr mesh_ptr (new Mesh);
					grid_struct.tsdf_ptr->generateMesh(mesh_ptr->vertices, mesh_ptr->triangles);
					MeshUtilities::transformMeshVertices(grid_struct.getExternalTSDFPose(), mesh_ptr->vertices);
					update_interface_->updateMesh("debug_clipping_grid_mesh", mesh_ptr);
				}
				cout << "debug_clipping pause..." << endl;
				cv::waitKey();
			}
		}

        /////
		// see if the clipping resulted in an empty box, in which case mark as merged
		// todo: should also probably eliminate from graph??
		{
			if (!grid_struct.tsdf_ptr->doesVolumeContainSurface()) {
				grid_struct.merged = true;
                if (params_.moving_volume_grid.debug_delete_edges_for_merged_volumes) {
                    boost::timer t;
                    // remove edges which touch this volume
                    std::vector<EdgeStruct> old_keyframe_edge_list = keyframe_edge_list_;
                    keyframe_edge_list_.clear();
                    BOOST_FOREACH(EdgeStruct const& e, old_keyframe_edge_list) {
                        if (e.index_1 != i) keyframe_edge_list_.push_back(e);
                    }
                    cout << "[TIMING] remove edges for merged volume " << i << ":" << t.elapsed() << endl;
                }
			}
		}

        //////
        // transfer keyframe associations to appropriate "world" block...
        // I think I can simply look at all 8 corners of this block, find their bounding box, and copy these keyframes...
        {
            std::vector<Eigen::Vector3f> corners = grid_struct.tsdf_ptr->getVolumeCorners(grid_struct.getExternalTSDFPose());
			Eigen::Vector3f bb_min = corners[0];
			Eigen::Vector3f bb_max = corners[0];
            BOOST_FOREACH(Eigen::Vector3f const& c, corners) {
				bb_min = bb_min.array().min(c.array()).matrix();
				bb_max = bb_max.array().max(c.array()).matrix();
            }
			// now get all bb corners
			std::vector<Eigen::Vector3f> bb_corners = getBoxCorners(bb_min.array(), bb_max.array(), Eigen::Affine3f::Identity());
			// finally for these corners, find associated world block, and add the keyframes for this grid_struct to that world block
			BOOST_FOREACH(Eigen::Vector3f const& c, bb_corners) {
				// could also use equivalent getGridCell();
				Eigen::Array3i world_block = EigenUtilities::floorArray3fToInt(c.array() / params_.grid.grid_size / params_.volume.cell_size);
				// only put in if this world block is inside the moving volume
				if ( (world_block >= moving_volume_block_offset_).all() && (world_block < moving_volume_block_offset_ + params_.moving_volume_grid.blocks_in_moving_volume).all() ) {
					BlockIndexT world_block_index = EigenUtilities::array3iToTuple(world_block);
					std::set<int> const& keyframes_for_grid = volume_to_keyframe_[i];
					std::set<int> & keyframes_for_world_block = block_index_to_keyframe_[world_block_index];
					std::copy(keyframes_for_grid.begin(), keyframes_for_grid.end(), std::inserter(keyframes_for_world_block, keyframes_for_world_block.end()));
				}
            }
        }

		// debug
		if (params_.loop_closure.debug_merge_show_points) {
			// show points in merged volume only in box
			{
				Eigen::Affine3f grid_struct_pose = grid_struct.getExternalTSDFPose();
				Eigen::Array3f box_min, box_max;
				grid_struct.tsdf_ptr->getBBForPose(grid_struct_pose, box_min, box_max);

				MeshVertexVectorPtr point_vertices(new MeshVertexVector);
				moving_tsdf_ptr_->getPrettyVoxelCenters(moving_tsdf_pose_, box_min, box_max, *point_vertices);
				update_interface_->updatePointCloud("debug_merge_show_points", point_vertices);
			}

			cout << "debug_merge_show_points pause..." << endl;
			cv::waitKey();

			// instead show moving mesh...will be slow
			{
				MeshPtr mesh_ptr (new Mesh);
				moving_tsdf_ptr_->generateMesh(mesh_ptr->vertices, mesh_ptr->triangles);
				MeshUtilities::transformMeshVertices(moving_tsdf_pose_, mesh_ptr->vertices);
				update_interface_->updateMesh("debug_merge_show_points_mesh", mesh_ptr);
			}
#if 0
			// moving points
			MeshVertexVectorPtr point_vertices(new MeshVertexVector);
			moving_tsdf_ptr_->getPrettyVoxelCenters(moving_tsdf_pose_, debug_show_empty, *point_vertices);
			update_interface_->updatePointCloud("debug_merge_show_points", point_vertices);
#endif
			cout << "debug_merge_show_points pause..." << endl;
			cv::waitKey();
		}

	}
}

bool ModelMovingVolumeGrid::loopClosure(Frame& frame)
{
	bool result = ModelGrid::loopClosure(frame);
	if (result) {
		mergeVolumesIntoMovingVolume();
		if (params_.loop_closure.debug_save_meshes) {
			MeshVertexVector vertex_list;
			TriangleVector triangle_list;
			generateMesh(vertex_list, triangle_list);
			fs::path save_file = params_.volume_modeler.output / (boost::format("%05d_after_loop_and_merge.ply") % getCameraListSize()).str();
			MeshUtilities::saveMesh(vertex_list, triangle_list, save_file);
		}
	}
	return result;
}
