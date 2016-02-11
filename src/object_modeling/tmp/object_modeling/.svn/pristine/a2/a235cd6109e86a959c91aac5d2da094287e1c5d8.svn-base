#include "model_single_volume.h"

#include "model_grid.h" // for merge

#include "EigenUtilities.h"

#include "MeshUtilities.h"

ModelSingleVolume::ModelSingleVolume(boost::shared_ptr<OpenCLAllKernels> all_kernels, VolumeModelerAllParams const& params, boost::shared_ptr<Alignment> alignment_ptr, RenderBuffers const& render_buffers)
	: ModelBase(all_kernels, params, alignment_ptr, render_buffers),
	tsdf_pose_(Eigen::Affine3f::Identity()) // redundant with reset?
{
	reset();
}

ModelSingleVolume::ModelSingleVolume(ModelSingleVolume const& other)
	: ModelBase(other),
	tsdf_pose_(other.tsdf_pose_),
	opencl_tsdf_ptr_(other.opencl_tsdf_ptr_->clone())
{
}

ModelSingleVolume* ModelSingleVolume::clone()
{
	return new ModelSingleVolume(*this);
}

void ModelSingleVolume::reset()
{
	ModelBase::reset();

    opencl_tsdf_ptr_.reset(new OpenCLTSDF(all_kernels_,
		params_.camera.focal[0], params_.camera.focal[1], 
		params_.camera.center[0], params_.camera.center[1], 
		params_.camera.size[0], params_.camera.size[1],
		params_.volume.cell_count.x(), params_.volume.cell_count.y(), params_.volume.cell_count.z(),
		params_.volume.cell_size,
		params_.volume.max_weight_icp,
		params_.volume.max_weight_color,
        params_.volume.use_most_recent_color,
            params_.volume.min_truncation_distance));

	tsdf_pose_ = Eigen::Affine3f::Identity();
}

void ModelSingleVolume::renderModel(
        const ParamsCamera & params_camera,
        const Eigen::Affine3f & model_pose,
        RenderBuffers & render_buffers)
{
    render_buffers.setSize(params_camera.size.x(), params_camera.size.y());
	render_buffers.resetAllBuffers();

    const int mask_value = 1;
    opencl_tsdf_ptr_->renderFrame(model_pose * tsdf_pose_,
        params_camera.focal, params_camera.center, params_camera.min_max_depth,
        false, mask_value,
		render_buffers);
}

void ModelSingleVolume::updateModel(
	Frame & frame,
	const Eigen::Affine3f & model_pose)
{
	// do this before base class because won't be empty after that
	if (isEmpty()) {
		tsdf_pose_ = getSingleVolumePoseForFirstFrame(frame);
	}

	ModelBase::updateModel(frame, model_pose);

	// can leave buffer_segments empty if which_segment = 0;
	// but could also use to for masking...
	ImageBuffer buffer_segments(all_kernels_->getCL());
	opencl_tsdf_ptr_->addFrame(model_pose * tsdf_pose_, frame.image_buffer_depth, frame.image_buffer_color, frame.image_buffer_segments, frame.image_buffer_add_depth_weights, frame.image_buffer_add_color_weights, 0);
}

void ModelSingleVolume::generateMesh(MeshVertexVector & vertex_list, TriangleVector & triangle_list)
{
	opencl_tsdf_ptr_->generateMesh(vertex_list, triangle_list);
	MeshUtilities::transformMeshVertices(tsdf_pose_, vertex_list);
}

void ModelSingleVolume::generateMeshAndValidity(MeshVertexVector & vertex_list, TriangleVector & triangle_list, std::vector<bool> & vertex_validity, std::vector<bool> & triangle_validity)
{
	opencl_tsdf_ptr_->generateMeshAndValidity(vertex_list, triangle_list, vertex_validity);
	MeshUtilities::getTriangleValidity(vertex_list, triangle_list, vertex_validity, triangle_validity);
	MeshUtilities::transformMeshVertices(tsdf_pose_, vertex_list);
}

void ModelSingleVolume::getBuffers(std::vector<float>& bufferDVector, std::vector<float>& bufferDWVector, std::vector<unsigned char>& bufferCVector, std::vector<float>& bufferCWVector)
{
	opencl_tsdf_ptr_->getAllBuffers(bufferDVector, bufferDWVector, bufferCVector, bufferCWVector);
}

void ModelSingleVolume::getBufferD(std::vector<float>& bufferDVector)
{
	opencl_tsdf_ptr_->getBufferD(bufferDVector);
}

void ModelSingleVolume::getBufferDW(std::vector<float>& bufferDWVector)
{
	opencl_tsdf_ptr_->getBufferDW(bufferDWVector);
}

void ModelSingleVolume::getBufferC(std::vector<unsigned char>& bufferCVector)
{
	opencl_tsdf_ptr_->getBufferC(bufferCVector);
}

void ModelSingleVolume::getBufferCW(std::vector<float>& bufferCWVector)
{
	opencl_tsdf_ptr_->getBufferCW(bufferCWVector);
}

void ModelSingleVolume::getBuffersLists(
	std::vector<boost::shared_ptr<std::vector<float> > >& bufferDVectors, 
	std::vector<boost::shared_ptr<std::vector<float> > > & bufferDWVectors, 
	std::vector<boost::shared_ptr<std::vector<unsigned char> > > & bufferCVectors, 
	std::vector<boost::shared_ptr<std::vector<float> > > & bufferCWVectors,
	std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list,
	std::vector<boost::shared_ptr<Eigen::Array3i> > & cell_counts_list)
{
	bufferDVectors.assign(1, boost::make_shared<std::vector<float> >());
	bufferDWVectors.assign(1, boost::make_shared<std::vector<float> >());
	bufferCVectors.assign(1, boost::make_shared<std::vector<unsigned char> >());
	bufferCWVectors.assign(1, boost::make_shared<std::vector<float> >());
	pose_list.assign(1, boost::shared_ptr<Eigen::Affine3f>(new Eigen::Affine3f));
	cell_counts_list.assign(1, boost::shared_ptr<Eigen::Array3i>(new Eigen::Array3i));

	opencl_tsdf_ptr_->getAllBuffers(*bufferDVectors.back(), *bufferDWVectors.back(), *bufferCVectors.back(), *bufferCWVectors.back());
	*pose_list.back() = tsdf_pose_;
	*cell_counts_list.back() = Eigen::Array3i(params_.volume.cell_count.x(), params_.volume.cell_count.y(), params_.volume.cell_count.z());
}

void ModelSingleVolume::deallocateBuffers()
{
	opencl_tsdf_ptr_->setTSDFState(TSDF_STATE_RAM);
}

void ModelSingleVolume::save(fs::path const& folder)
{
	ModelBase::save(folder);

	opencl_tsdf_ptr_->save(folder);

	{
		fs::path filename = folder / "model_single_volume_tsdf_pose.txt";
		std::ofstream file (filename.string().c_str());
		file << EigenUtilities::transformToString(tsdf_pose_) << endl;
	}
}

void ModelSingleVolume::load(fs::path const& folder)
{
	ModelBase::load(folder);

	opencl_tsdf_ptr_->load(folder);
	
	{
		fs::path filename = folder / "model_single_volume_tsdf_pose.txt";
		std::ifstream file(filename.string().c_str());
		std::string line;
		std::getline(file, line);
		tsdf_pose_ = EigenUtilities::stringToTransform(line);
	}
}

void ModelSingleVolume::setMaxWeightInVolume(float new_weight)
{
	opencl_tsdf_ptr_->setMaxWeightInVolume(new_weight);
}

void ModelSingleVolume::mergeOther(ModelSingleVolume & other, Eigen::Affine3f const& relative_pose)
{
	Eigen::Affine3f relative_pose_arg = other.tsdf_pose_.inverse() * relative_pose * tsdf_pose_;
	opencl_tsdf_ptr_->addVolume(*other.opencl_tsdf_ptr_, relative_pose_arg);
}

void ModelSingleVolume::mergeOther(ModelGrid & other, Eigen::Affine3f const& relative_pose)
{
	//Eigen::Affine3f relative_pose_arg = other.tsdf_pose.inverse() * relative_pose * tsdf_pose;

	// add in the grids with our knowledge of them (ugly)

	for (size_t i = 0; i < other.grid_list_.size(); ++i) {
		ModelGrid::GridStruct & grid_struct = *other.grid_list_[i];
		if (!grid_struct.active) continue;

		Eigen::Affine3f relative_pose_arg = (grid_struct.pose_external * grid_struct.pose_tsdf).inverse() * relative_pose * tsdf_pose_;
		opencl_tsdf_ptr_->addVolumeSphereTest(*grid_struct.tsdf_ptr, relative_pose_arg);

		other.deallocateAsNeeded();
	}
}

Eigen::Affine3f ModelSingleVolume::getPose() const
{
	return tsdf_pose_;
}

void ModelSingleVolume::setPose(Eigen::Affine3f const& pose)
{
	tsdf_pose_ = pose;
}

void ModelSingleVolume::debugSetSphere(float radius)
{
	opencl_tsdf_ptr_->setVolumeToSphere(radius);
}


void ModelSingleVolume::setValueInSphere(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Vector3f const& center, float radius)
{
	Eigen::Vector3f center_in_tsdf = tsdf_pose_.inverse() * center;
	opencl_tsdf_ptr_->setValueInSphere(d_value, dw_value, c_value, cw_value, center_in_tsdf, radius);
}

void ModelSingleVolume::setValueInBox(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Affine3f const& pose)
{
	Eigen::Affine3f pose_in_tsdf = tsdf_pose_.inverse() * pose;
	opencl_tsdf_ptr_->setValueInBox(d_value, dw_value, c_value, cw_value, pose_in_tsdf);
}

void ModelSingleVolume::getBoundingMesh(MeshVertexVector & vertex_list, TriangleVector & triangle_list)
{
	Eigen::Vector4ub color(0,0,255,0);
	opencl_tsdf_ptr_->getBoundingMesh(tsdf_pose_, color, vertex_list, triangle_list);
}

void ModelSingleVolume::getBoundingLines(MeshVertexVector & vertex_list)
{
	Eigen::Vector4ub color(0,0,255,0);
	opencl_tsdf_ptr_->getBoundingLines(tsdf_pose_, color, vertex_list);
}

void ModelSingleVolume::getNonzeroVolumePointCloud(MeshVertexVector & vertex_list)
{
	opencl_tsdf_ptr_->getPrettyVoxelCenters(tsdf_pose_, vertex_list);
}

void ModelSingleVolume::debugAddVolume()
{
    boost::shared_ptr<OpenCLTSDF> new_tsdf_ptr (new OpenCLTSDF(all_kernels_,
		params_.camera.focal[0], params_.camera.focal[1], 
		params_.camera.center[0], params_.camera.center[1], 
		params_.camera.size[0], params_.camera.size[1],
		params_.volume.cell_count.x(), params_.volume.cell_count.y(), params_.volume.cell_count.z(),
		params_.volume.cell_size,
		params_.volume.max_weight_icp,
		params_.volume.max_weight_color,
        params_.volume.use_most_recent_color,
            params_.volume.min_truncation_distance));

	new_tsdf_ptr->addVolume(*opencl_tsdf_ptr_, Eigen::Affine3f::Identity());

	opencl_tsdf_ptr_ = new_tsdf_ptr;
}

void ModelSingleVolume::refreshUpdateInterface()
{
	if (update_interface_) {
		{
			MeshVertexVectorPtr bounding_lines_ptr (new MeshVertexVector);
			getBoundingLines(*bounding_lines_ptr);
            update_interface_->updateLines(params_.glfw_keys.volumes_all, bounding_lines_ptr);
		}
	}
}
