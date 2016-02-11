#include "model_base.h"

ModelBase::ModelBase(boost::shared_ptr<OpenCLAllKernels> all_kernels, VolumeModelerAllParams const& params, boost::shared_ptr<Alignment> alignment_ptr, RenderBuffers const& render_buffers)
	: all_kernels_(all_kernels),
	params_(params),
	alignment_ptr_(alignment_ptr),
	render_buffers_(render_buffers)
{
}

ModelBase::ModelBase(ModelBase const& other)
	: all_kernels_(other.all_kernels_),
	params_(other.params_),
	alignment_ptr_(other.alignment_ptr_),
	render_buffers_(other.render_buffers_)
{
	camera_list_.resize(other.camera_list_.size());
	for (int i = 0 ; i < other.camera_list_.size(); ++i) {
		camera_list_[i].reset(new CameraStruct(*other.camera_list_[i]));
	}
}

void ModelBase::reset()
{
	camera_list_.clear();
}

bool ModelBase::isEmpty() const
{
	return camera_list_.empty();
}

size_t ModelBase::getCameraListSize() const
{
	return camera_list_.size();
}

void ModelBase::setUpdateInterface(boost::shared_ptr<UpdateInterface> update_interface_ptr)
{
	update_interface_ = update_interface_ptr;
}

void ModelBase::setPickPixel(boost::shared_ptr<PickPixel> pick_pixel_ptr)
{
	pick_pixel_ = pick_pixel_ptr;
}

void ModelBase::setDebugStringPrefix(const std::string & s)
{
	debug_string_prefix_ = s;
}

std::map<std::string, cv::Mat> const& ModelBase::getDebugImages() const
{
	return debug_images_;
}

void ModelBase::debugAddVolume() {
	cout << "Warning: model_base debugAddVolume() does nothing" << endl;
}

void ModelBase::prepareForRenderCurrent()
{
	// nothing
}

void ModelBase::save(fs::path const& folder)
{
	// camera_list
	// format is number of cameras, followed by all members
	// should really encapulate this in camera_struct
	{
		fs::path filename = folder / "camera_list.txt";
		std::fstream file(filename.string().c_str(), std::ios::out);
		file << camera_list_.size() << endl;
		for (int i = 0; i < (int)camera_list_.size(); ++i) {
			CameraStruct & cam = *camera_list_[i];
			file << EigenUtilities::transformToString(cam.pose) << endl;
			file << EigenUtilities::transformToString(cam.original_pose) << endl;
			file << cam.ros_timestamp << endl;
		}
	}
}

void ModelBase::load(fs::path const& folder)
{
	// camera_list
	// format is number of cameras, followed by all members
	// should really encapulate this in camera_struct
	{
		fs::path filename = folder / "camera_list.txt";
		std::fstream file(filename.string().c_str(), std::ios::in);
		std::string s;
		int camera_list_size;
		file >> camera_list_size;
		std::getline(file, s); // eat line
		camera_list_.resize(camera_list_size);
		for (int i = 0; i < (int)camera_list_.size(); ++i) {
			camera_list_[i].reset(new CameraStruct(Eigen::Affine3f::Identity(), ROSTimestamp()));
			CameraStruct & cam = *camera_list_[i];
			std::getline(file, s);
			cam.pose = EigenUtilities::stringToTransform(s);
			std::getline(file, s);
			cam.original_pose = EigenUtilities::stringToTransform(s);
			std::getline(file, s);
			cam.ros_timestamp = ROSTimestamp(s);
		}
	}
}

void ModelBase::updateModel(
	Frame & frame,
	const Eigen::Affine3f & model_pose)
{
	camera_list_.push_back(CameraStructPtr(new CameraStruct(model_pose.inverse(), frame.ros_timestamp)));
}

Eigen::Affine3f ModelBase::getLastCameraPose() const {
	if (camera_list_.empty()) {
		cout << "warning: getLastCameraPose on empty camera_list_" << endl;
		return Eigen::Affine3f::Identity();
	}
	else {
		return camera_list_.back()->pose;
	}
}

bool ModelBase::loopClosure(Frame& frame)
{
	// don't have to say this...
	cout << "Warning: ModelBase does not support loop closure" << endl;
	return false;
}

void ModelBase::getAllCameras(std::vector<CameraStructPtr> & cameras_list)
{
	cameras_list.clear();

	for (std::vector<CameraStructPtr>::const_iterator iter = camera_list_.begin(); iter != camera_list_.end(); ++iter) {
		cameras_list.push_back(CameraStructPtr(new CameraStruct(**iter)));
	}
}

bool ModelBase::getCameraPose(size_t i, Eigen::Affine3f & result_pose)
{
	if (i < camera_list_.size()) {
		result_pose = camera_list_[i]->pose;
		return true;
	}

	return false;
}

void ModelBase::refreshUpdateInterface()
{
	if (update_interface_) {
		// view pose change?
		if (params_.volume_modeler.update_interface_view_pose) {
			update_interface_->updateViewPose(camera_list_.back()->pose);
		}

        //  all cameras
        {
			UpdateInterface::PoseListPtrT pose_list_ptr(new UpdateInterface::PoseListT);
			for (int i = 0; i < camera_list_.size(); ++i) {
				pose_list_ptr->push_back(camera_list_[i]->pose);
			}
            update_interface_->updateCameraList(params_.glfw_keys.cameras_all, pose_list_ptr);
		}
	}
}

void ModelBase::setMaxWeightInVolume(float new_weight)
{
	// nothing
}

void ModelBase::getBuffersLists(
	std::vector<boost::shared_ptr<std::vector<float> > >& bufferDVectors,
	std::vector<boost::shared_ptr<std::vector<float> > > & bufferDWVectors,
	std::vector<boost::shared_ptr<std::vector<unsigned char> > > & bufferCVectors,
	std::vector<boost::shared_ptr<std::vector<float> > > & bufferCWVectors,
	std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list,
	std::vector<boost::shared_ptr<Eigen::Array3i> > & cell_counts_list)
{
	// nothing??
}

void ModelBase::setValueInSphere(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Vector3f const& center, float radius)
{
	// nothing
}

void ModelBase::setValueInBox(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Affine3f const& pose)
{
	// nothing
}

void ModelBase::getNonzeroVolumePointCloud(MeshVertexVector & vertex_list)
{
	// nothing
}

void ModelBase::generateAllMeshes(std::vector<std::pair<std::string, MeshPtr> > & names_and_meshes)
{
	// default is just the first mesh and name "mesh"
	MeshPtr default_mesh(new Mesh);
	this->generateMesh(default_mesh->vertices, default_mesh->triangles);
	names_and_meshes.clear();
	names_and_meshes.push_back(std::make_pair("mesh", default_mesh)); 
}

Eigen::Affine3f ModelBase::getSingleVolumePoseForFirstFrame(Frame& frame)
{
	Eigen::Affine3f result(Eigen::Affine3f::Identity());
	if (params_.volume_modeler.first_frame_centroid) {
		// need to find centroid of first frame...sigh...
		// pull image_buffer_points off the GPU??
		std::vector<float> image_buffer_points_vector(frame.image_buffer_points.getSizeBytes() / sizeof(float));
		frame.image_buffer_points.getBufferWrapper().readToFloatVector(image_buffer_points_vector);
		std::vector<Eigen::Array4f, Eigen::aligned_allocator<Eigen::Array4f> > points;
		for (size_t i = 0; i < image_buffer_points_vector.size(); i += 4) {
			float const& test = image_buffer_points_vector[i];
			if (test == test) {
				points.push_back(Eigen::Array4f::Map(&image_buffer_points_vector[i]));
			}
		}
		Eigen::Array4f sum(0,0,0,0);
		for (size_t i = 0; i < points.size(); ++i) {
			sum += points[i];
		}
		Eigen::Array4f centroid = sum / (float)points.size();
		Eigen::Vector3f centroid_vector = centroid.head<3>();
		Eigen::Vector3f volume_centering_vector(
			-params_.volume.cell_count.x() / 2 * params_.volume.cell_size,
			-params_.volume.cell_count.y() / 2 * params_.volume.cell_size,
			-params_.volume.cell_count.z() / 2 * params_.volume.cell_size);
		result.pretranslate(centroid_vector + volume_centering_vector);
	}
	else if (params_.volume_modeler.first_frame_origin) {
		// useful for generated data
		Eigen::Vector3f t(
			-params_.volume.cell_count.x() / 2 * params_.volume.cell_size,
			-params_.volume.cell_count.y() / 2 * params_.volume.cell_size,
			-params_.volume.cell_count.z() / 2 * params_.volume.cell_size);
		result.pretranslate(t);
	}
	else {
		Eigen::Vector3f t(
			-params_.volume.cell_count.x() / 2 * params_.volume.cell_size,
			-params_.volume.cell_count.y() / 2 * params_.volume.cell_size,
			params_.camera.min_max_depth[0]);
		result.pretranslate(t);
	}
	return result;
}

void ModelBase::saveGraphs(const fs::path & folder)
{
    if (!fs::exists(folder) && !fs::create_directories(folder)) {
        cout << "Warning: failed to save graphs to: " << folder << endl;
        return;
    }

    // what format, then...
    cout << "Warning: saveGraphs not implemented" << endl;
}
