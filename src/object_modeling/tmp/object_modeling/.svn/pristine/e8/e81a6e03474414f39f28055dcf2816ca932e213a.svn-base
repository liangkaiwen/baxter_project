#include "volume_modeler.h"

#include "EigenUtilities.h"

#include "MeshUtilities.h"

#include "opencv_utilities.h"

#include "model_single_volume.h"
#include "model_grid.h"
#include "model_patch.h"
#include "model_moving_volume_grid.h"
#include "model_histogram.h"
#include "model_k_means.h"

// for parts of load and save
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>

#include "KernelDepthImageToPoints.h"


VolumeModeler::VolumeModeler(boost::shared_ptr<OpenCLAllKernels> all_kernels, VolumeModelerAllParams const& params)
	: all_kernels_(all_kernels),
    render_buffers_(all_kernels),
	params_(params),
	frustum_(params.camera, Eigen::Affine3f::Identity())
{
	const bool opencl_debug = false;
	const bool opencl_fastmath = false;

	// normals kernels
	opencl_normals_ptr_.reset(new OpenCLNormals(all_kernels_));

	// init render_buffers
	render_buffers_.setSize(params_.camera.size.x(), params_.camera.size.y());
	render_buffers_.resetAllBuffers();

	// alignment pointer now:
	alignment_ptr_.reset (new Alignment(all_kernels_, params_.alignment));

	// feature matcher:
	features_ptr_.reset (new FeatureMatching(params_.camera, params_.features)); 

	// mask object:
	mask_object_.reset (new MaskObject(params.mask_object));

	//reset();
	allocateMainModel();
	allocateModelList();
}

VolumeModeler::VolumeModeler(VolumeModeler const& other)
	: 	params_(other.params_),
	all_kernels_(other.all_kernels_),
	opencl_normals_ptr_(other.opencl_normals_ptr_),
	render_buffers_(other.render_buffers_),
	frustum_(other.frustum_),
	alignment_ptr_(other.alignment_ptr_),
	features_ptr_(other.features_ptr_),
	update_interface_(other.update_interface_),
	debug_images_(other.debug_images_),
	previous_frame_pose_(other.previous_frame_pose_),
	previous_frame_keypoints_(other.previous_frame_keypoints_),
	previous_frame_image_(other.previous_frame_image_),
	mask_object_(other.mask_object_)
{
	model_ptr_.reset(other.model_ptr_->clone());
	model_ptr_list_.clear();
	BOOST_FOREACH(const boost::shared_ptr<ModelBase> & m, other.model_ptr_list_) {
		model_ptr_list_.push_back(boost::shared_ptr<ModelBase>(m->clone()));
	}
	// todo: should I deep copy things like render buffers?
}


void VolumeModeler::getParams(VolumeModelerAllParams & params) const
{
	params = params_;
}

// static utility
ModelType VolumeModeler::modelTypeFromString(std::string model_type_string)
{
	if (model_type_string.empty() || model_type_string == "none") {
		return MODEL_NONE;
	}
	if (model_type_string == "single") {
		return MODEL_SINGLE_VOLUME;
	}
	else if (model_type_string == "grid") {
		return MODEL_GRID;
	}
	else if (model_type_string == "patch") {
		return MODEL_PATCH;
	}
	else if (model_type_string == "moving") {
		return MODEL_MOVING_VOLUME_GRID;
	}
	else if (model_type_string == "histogram") {
		return MODEL_HISTOGRAM;
	}
	else if (model_type_string == "kmeans") {
		return MODEL_K_MEANS;
	}
	else {
		return MODEL_ERROR;
	}
}

boost::shared_ptr<ModelBase> VolumeModeler::allocateNewModel(ModelType model_type)
{
	if (model_type == MODEL_NONE) {
		return boost::shared_ptr<ModelBase>();
	}
	if (model_type == MODEL_SINGLE_VOLUME) {
		return boost::shared_ptr<ModelBase>(new ModelSingleVolume(all_kernels_, params_, alignment_ptr_, render_buffers_));
	}
	else if (model_type == MODEL_GRID) {
		return boost::shared_ptr<ModelBase>(new ModelGrid(all_kernels_, params_, alignment_ptr_, render_buffers_));
	}
	else if (model_type == MODEL_PATCH) {
		return boost::shared_ptr<ModelBase>(new ModelPatch(all_kernels_, params_, alignment_ptr_, render_buffers_));
	}
    else if (model_type == MODEL_MOVING_VOLUME_GRID) {
        return boost::shared_ptr<ModelBase>(new ModelMovingVolumeGrid(all_kernels_, params_, alignment_ptr_, render_buffers_));
    }
    else if (model_type == MODEL_HISTOGRAM) {
        return boost::shared_ptr<ModelBase>(new ModelHistogram(all_kernels_, params_, alignment_ptr_, render_buffers_));
    }
	else if (model_type == MODEL_K_MEANS) {
		return boost::shared_ptr<ModelBase>(new ModelKMeans(all_kernels_, params_, alignment_ptr_, render_buffers_));
	}
	else {
		cout << "Unknown model type!" << endl;
		throw std::runtime_error("unknown model_type");
	}
}

void VolumeModeler::allocateMainModel()
{
	model_ptr_ = allocateNewModel(params_.volume_modeler.model_type);
}

void VolumeModeler::allocateModelList()
{
	model_ptr_list_.clear();
	boost::shared_ptr<ModelBase> model_1 = allocateNewModel(params_.volume_modeler.model_type_1);
	if (model_1) {
		model_ptr_list_.push_back(model_1);
		model_ptr_list_.back()->setDebugStringPrefix("mt1_");
	}
	else return; // must have 1 for 2?
	boost::shared_ptr<ModelBase> model_2 = allocateNewModel(params_.volume_modeler.model_type_2);
	if (model_2) {
		model_ptr_list_.push_back(model_2);
		model_ptr_list_.back()->setDebugStringPrefix("mt2_");
	}
}


void VolumeModeler::reset()
{
	model_ptr_->reset();
	BOOST_FOREACH(boost::shared_ptr<ModelBase> & m, model_ptr_list_) {
		m->reset();
	}
	previous_frame_pose_ = Eigen::Affine3f::Identity();
	previous_frame_keypoints_.reset();
	previous_frame_image_ = cv::Mat();
}

bool VolumeModeler::isFirstFrame() const
{
	return model_ptr_->isEmpty();
}

void VolumeModeler::ensureImageBuffers(Frame & frame)
{
	if (frame.image_buffer_color.getSizeBytes() == 0) {
		frame.copyToImageBuffers();
	}
}

void VolumeModeler::ensurePointsBuffer(Frame & frame)
{
	ensureImageBuffers(frame);

	if (frame.image_buffer_points.getSizeBytes() == 0) {
		KernelDepthImageToPoints _KernelDepthImageToPoints(*all_kernels_);
		_KernelDepthImageToPoints.runKernel(frame.image_buffer_depth, frame.image_buffer_points, params_.camera.focal, params_.camera.center);
	}
}

void VolumeModeler::ensureNormalsBuffer(Frame & frame)
{
	ensurePointsBuffer(frame);

	if (frame.image_buffer_normals.getSizeBytes() == 0) {
		opencl_normals_ptr_->computeNormalsWithBuffers(frame.image_buffer_points, params_.normals.max_depth_sigmas, params_.normals.smooth_iterations, frame.image_buffer_normals);
	}
}

void VolumeModeler::getPoints(Frame & frame, std::vector<float> & points)
{
	ensurePointsBuffer(frame);

	points.resize(frame.mat_depth.total() * 4);
	frame.image_buffer_points.getBufferWrapper().readToFloatVector(points);
}

void VolumeModeler::getNormals(Frame & frame, std::vector<float> & normals)
{
	ensureNormalsBuffer(frame);

	normals.resize(frame.mat_depth.total() * 4);
	frame.image_buffer_normals.getBufferWrapper().readToFloatVector(normals);
}

bool VolumeModeler::alignAndAddFrame(Frame & frame)
{
	Eigen::Affine3f camera_pose;
	bool align_success = alignFrame(frame, camera_pose);
	if (align_success) {
		addFrame(frame, camera_pose);
	}
	return align_success;
}

bool VolumeModeler::alignFrame(Frame& frame, Eigen::Affine3f& camera_pose_result)
{
	// would also need to do in add frame if using pre-existing poses
	// should really be in a prepare frame with an ensure...and only do once.
	// but to do that you'd need to know when you'd already applied it
	if (params_.mask_object.mask_object) {
		cv::Rect object_rect;
		cv::Mat object_mask;
		mask_object_->getObjectMask(frame, object_rect, object_mask);
		frame.applyMaskToDepth(object_mask);
		debug_images_["object_mask"] = object_mask;
	}

	ensureNormalsBuffer(frame);

	// compute features if we're using them
	if (params_.volume_modeler.use_features) {
		features_ptr_->addFeaturesForFrameIfNeeded(frame);
	}

	Eigen::Affine3f model_pose = getLastCameraPose().inverse();
	bool do_add_frame = false;
	if (isFirstFrame()) {
		do_add_frame = true;
	}
	else {
		// features first
		if (params_.volume_modeler.use_features) {
			if (!previous_frame_keypoints_) {
				cout << "Warning: use_features but previous_frame_keypoints_ is NULL" << endl;
			}
			else {
				boost::timer t_features;
				std::vector<cv::DMatch> matches;
				features_ptr_->matchDescriptors(*frame.keypoints, *previous_frame_keypoints_, matches);
				Eigen::Affine3f feature_relative_pose;
				std::vector<cv::DMatch> inlier_matches;
				bool ransac_success = features_ptr_->ransac(*frame.keypoints, *previous_frame_keypoints_, matches, feature_relative_pose, inlier_matches);

				if (!params_.volume_modeler.command_line_interface) {
                    // can re-enable for feature match image
#if 0
					cv::Mat image_matches;
					cv::drawMatches(frame.mat_color, frame.keypoints->keypoints, previous_frame_image_, previous_frame_keypoints_->keypoints, inlier_matches, image_matches);
					debug_images_["image_matches"] = image_matches;
#endif
				}

				if (ransac_success) {
					Eigen::Affine3f new_camera_pose_features = previous_frame_pose_ * feature_relative_pose;
					model_pose = new_camera_pose_features.inverse();
					do_add_frame = true;
				}
				else {
					cout << "Warning: feature alignment failed with inlier count: " << inlier_matches.size() << endl;
				}
				if (params_.volume_modeler.verbose) cout << "TIME align with features: " << t_features.elapsed() << endl;
			}
		}

		// relative alignment
		boost::timer t_render;
        model_ptr_->prepareForRenderCurrent();
		render(model_pose.inverse());
		if (params_.volume_modeler.verbose) cout << "TIME render: " << t_render.elapsed() << endl;

		Eigen::Affine3f initial_relative_pose = Eigen::Affine3f::Identity();
		Eigen::Affine3f result_pose = Eigen::Affine3f::Identity();

		boost::timer t_align;
		std::vector<int> result_iterations;
		bool alignment_success = alignment_ptr_->align(
				frame.image_buffer_color,
				frame.image_buffer_points,
				frame.image_buffer_normals,
				frame.image_buffer_align_weights,
				render_buffers_.getImageBufferColorImage(),
				render_buffers_.getImageBufferPoints(),
				render_buffers_.getImageBufferNormals(),
				render_buffers_.getImageBufferMask(),
                params_.camera,
				model_pose,
				initial_relative_pose,
				result_pose,
				result_iterations);
		if (params_.volume_modeler.verbose) {
			cout << "result_iterations: ";
			for (int i = 0; i < result_iterations.size(); ++i) cout << result_iterations[i] << " ";
			cout << endl;
		}
		if (alignment_success) {
			model_pose = result_pose;
			do_add_frame = true;
		}
		if (params_.volume_modeler.verbose) cout << "TIME dense alignment: " << t_align.elapsed() << endl;
	}

	if (do_add_frame) {
		camera_pose_result = model_pose.inverse();
		previous_frame_pose_ = camera_pose_result;
		previous_frame_keypoints_ = frame.keypoints; // safe even if no features
		previous_frame_image_ = frame.mat_color_bgra; // for showing matches in ransac
	}

	return do_add_frame;
}

void VolumeModeler::addFrame(Frame& frame, Eigen::Affine3f const& camera_pose)
{
	boost::timer t_add;

	//ensurePointsBuffer(frame);
	ensureNormalsBuffer(frame); // at least patch volumes need normals...you already needed them for align

	if (params_.volume_modeler.set_color_weights) {
		std::map<std::string, cv::Mat> set_color_weight_debug = frame.setColorWeights(params_.volume_modeler.max_edge_sigmas, params_.volume_modeler.max_distance_transform);
		frame.copyAddColorWeightsToImageBuffer(); // redundant but necessary to make sure we replace whatever crap is there already

		if (params_.volume_modeler.set_color_weights_debug_images) {
			for (std::map<std::string, cv::Mat>::const_iterator iter = set_color_weight_debug.begin(); iter != set_color_weight_debug.end(); ++iter) {
				debug_images_[iter->first] = iter->second;
			}
		}
	}

	// make sure pick_pixel is up to date here?
	if (pick_pixel_) {
		pick_pixel_->setMat(frame.mat_color_bgra);
	}

	// actually update the model
	Eigen::Affine3f model_pose = camera_pose.inverse();
	model_ptr_->updateModel(frame, model_pose);

	BOOST_FOREACH(boost::shared_ptr<ModelBase> & m, model_ptr_list_) {
		m->updateModel(frame, model_pose);
	}

	// end of function
	if (params_.volume_modeler.verbose) cout << "TIME add frame: " << t_add.elapsed() << endl;

	// update 3d viewer
	if (update_interface_) {
		boost::timer t_update_interface;

		// the points from the input frame
		{
			cv::Mat points_mat = frame.image_buffer_points.getMat();
			// put in mesh vertices
			MeshVertexVectorPtr point_vertices(new MeshVertexVector);
			for (int row = 0; row < points_mat.rows; ++row) {
				for (int col = 0; col < points_mat.cols; ++col) {
					// only valid vertices?
					cv::Vec4f & p = points_mat.at<cv::Vec4f>(row,col);
					if (p[2] > 0) {
						point_vertices->push_back(MeshVertex());
						MeshVertex & v = point_vertices->back();
						//v.p = Eigen::Vector4f::Map(&points_mat.at<float>(row,col));
						v.p[0] = p[0];
						v.p[1] = p[1];
						v.p[2] = p[2];
						v.p[3] = p[3];
						// transform point
						v.p = camera_pose * v.p;

						cv::Vec4b & c = frame.mat_color_bgra.at<cv::Vec4b>(row,col);
						v.c[0] = c[0];
						v.c[1] = c[1];
						v.c[2] = c[2];
						v.c[3] = c[3];
					}
				}
			}
            update_interface_->updatePointCloud(params_.glfw_keys.input_cloud, point_vertices);
		}

		// optionally (and slowly) the non-zero points from the volume
		// could move this into model_ptr refresh
		if (params_.volume_modeler.debug_show_nonzero_voxels) {
			MeshVertexVectorPtr point_vertices(new MeshVertexVector);
			model_ptr_->getNonzeroVolumePointCloud(*point_vertices);
			update_interface_->updatePointCloud("debug_show_nonzero_voxels", point_vertices);
		}

		// frustum
		{
			Frustum frustum(params_.camera, camera_pose);
			std::vector<Eigen::Vector3f> frustum_lineset_points = frustum.getLineSetPoints();
			MeshVertexVectorPtr frustum_vertices(new MeshVertexVector);
			for (int i = 0; i < (int)frustum_lineset_points.size(); ++i) {
				frustum_vertices->push_back(MeshVertex());
				MeshVertex & v = frustum_vertices->back();
				v.p.head<3>() = frustum_lineset_points[i];
				v.p[3] = 1;
				v.c[0] = v.c[1] = v.c[2] = 255;
			}
            update_interface_->updateLines(params_.glfw_keys.frustum, frustum_vertices);

			// also camera with frustum?
			UpdateInterface::PoseListPtrT frustum_camera(new UpdateInterface::PoseListT);
			frustum_camera->push_back(camera_pose);
            update_interface_->updateCameraList(params_.glfw_keys.frustum, frustum_camera);
            update_interface_->updateScale(params_.glfw_keys.frustum, 2);
            update_interface_->updateColor(params_.glfw_keys.frustum, Eigen::Array4ub(255,255,255,255));
		}

        refreshUpdateInterfaceForModels();

		if (params_.volume_modeler.verbose) cout << "TIME update_interface_ (avoidable): " << t_update_interface.elapsed() << endl;
	} // update_interface
}

bool VolumeModeler::doesModelSupportLoopClosure() const
{
	// currently model grid or subclasses
	ModelGrid* model_grid = dynamic_cast<ModelGrid*>(model_ptr_.get());
	return (bool)model_grid;
}

bool VolumeModeler::loopClosure(Frame & frame)
{
	if (!doesModelSupportLoopClosure()) {
		cout << "Warning: model does not support loop closure" << endl;
		return false;
	}

	return model_ptr_->loopClosure(frame);
}

void VolumeModeler::saveFullCommandLine(fs::path save_folder)
{
	fs::path filename = save_folder/"full_command_line.txt";
	std::ofstream ofs(filename.string().c_str());
	ofs << params_.volume_modeler.full_command_line << endl;
}

bool VolumeModeler::generateAndSaveMesh(fs::path save_file)
{
	MeshVertexVector vertex_list;
	TriangleVector triangle_list;
	generateMesh(vertex_list, triangle_list);
	return saveMesh(vertex_list, triangle_list, save_file);
}

bool VolumeModeler::generateAndSaveAllMeshes(fs::path save_folder)
{
	if (!fs::is_directory(save_folder) && !fs::create_directories(save_folder)) {
		cout << "failed to create directory: " << save_folder << endl;
		return false;
	}

	bool success = true;

	// save full command line
	saveFullCommandLine(save_folder);

	// use new "all meshes" function on model
	std::vector<std::pair<std::string, MeshPtr> > names_and_meshes;
	model_ptr_->generateAllMeshes(names_and_meshes);
	typedef std::pair<std::string, MeshPtr> Pair;
	BOOST_FOREACH(Pair & p, names_and_meshes) {
		std::string filename = (boost::format("%s.ply") % p.first).str();
        bool success_this = saveMesh(p.second->vertices, p.second->triangles, save_folder / filename);
        if (success_this) cout << "saveMesh: " << filename << endl;
        else cout << "FAIL saveMesh: " << filename << endl;
        success = success && success_this;
	}

	// mesh as well for other models?
	// really should do this whole function on all meshes (extra layer)
	for (size_t i = 0; i < model_ptr_list_.size(); ++i) {
		// make subfolder
		fs::path save_folder_for_model = save_folder / (boost::format("model_%d") % i).str();
		if (!fs::is_directory(save_folder_for_model) && !fs::create_directories(save_folder_for_model)) {
			cout << "failed to create directory: " << save_folder_for_model << endl;
			return false;
		}

		std::vector<std::pair<std::string, MeshPtr> > names_and_meshes;
		model_ptr_list_[i]->generateAllMeshes(names_and_meshes);
		typedef std::pair<std::string, MeshPtr> Pair;
		BOOST_FOREACH(Pair & p, names_and_meshes) {
			std::string filename = (boost::format("%s.ply") % p.first).str();
            bool success_this = saveMesh(p.second->vertices, p.second->triangles, save_folder_for_model / filename);
            if (success_this) cout << "saveMesh: " << filename << endl;
            else cout << "FAIL saveMesh: " << filename << endl;
            success = success && success_this;
		}
    }

	return success;
}

void VolumeModeler::generateMesh(MeshVertexVector & vertex_list, TriangleVector & triangle_list)
{
	model_ptr_->generateMesh(vertex_list, triangle_list);
}

void VolumeModeler::generateMeshAndValidity(MeshVertexVector & vertex_list, TriangleVector & triangle_list, std::vector<bool> & vertex_validity, std::vector<bool> & triangle_validity)
{
	model_ptr_->generateMeshAndValidity(vertex_list, triangle_list, vertex_validity, triangle_validity);
}

void VolumeModeler::generateMesh(
	const std::vector<boost::shared_ptr<std::vector<float> > > & bufferDVectors, 
	const std::vector<boost::shared_ptr<std::vector<float> > > & bufferDWVectors, 
	const std::vector<boost::shared_ptr<std::vector<unsigned char> > > & bufferCVectors, 
	const std::vector<boost::shared_ptr<std::vector<float> > > & bufferCWVectors,
	const std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list,
	const std::vector<boost::shared_ptr<Eigen::Array3i> > & cell_counts_list,
	MeshVertexVector & vertex_list,
	TriangleVector & triangle_list)
{
	vertex_list.clear();
	triangle_list.clear();

	for (size_t i = 0; i < bufferDVectors.size(); ++i) {
		MeshVertexVector this_vertex_list;
		TriangleVector this_triangle_list;
		MeshUtilities::generateMesh(*bufferDVectors[i], *bufferDWVectors[i], *bufferCVectors[i], *cell_counts_list[i], params_.volume.cell_size, this_vertex_list, this_triangle_list);
		MeshUtilities::transformMeshVertices(*pose_list[i], this_vertex_list);
		MeshUtilities::appendMesh(vertex_list, triangle_list, this_vertex_list, this_triangle_list);
	}

}

void VolumeModeler::generateMeshAndValidity(
	const std::vector<boost::shared_ptr<std::vector<float> > > & bufferDVectors, 
	const std::vector<boost::shared_ptr<std::vector<float> > > & bufferDWVectors, 
	const std::vector<boost::shared_ptr<std::vector<unsigned char> > > & bufferCVectors, 
	const std::vector<boost::shared_ptr<std::vector<float> > > & bufferCWVectors,
	const std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list,
	const std::vector<boost::shared_ptr<Eigen::Array3i> > & cell_counts_list,
	MeshVertexVector & vertex_list,
	TriangleVector & triangle_list,
	std::vector<bool> & vertex_validity,
	std::vector<bool> & triangle_validity)
{
	vertex_list.clear();
	triangle_list.clear();
	vertex_validity.clear();
	triangle_validity.clear();

	for (size_t i = 0; i < bufferDVectors.size(); ++i) {
		MeshVertexVector this_vertex_list;
		TriangleVector this_triangle_list;
		std::vector<bool> this_vertex_validity;
		std::vector<bool> this_triangle_validity;
		MeshUtilities::generateMeshAndValidity(*bufferDVectors[i], *bufferDWVectors[i], *bufferCVectors[i], *cell_counts_list[i], params_.volume.cell_size, this_vertex_list, this_triangle_list, this_vertex_validity);
		MeshUtilities::getTriangleValidity(this_vertex_list, this_triangle_list, this_vertex_validity, this_triangle_validity);
		MeshUtilities::transformMeshVertices(*pose_list[i], this_vertex_list);
		MeshUtilities::appendMesh(vertex_list, triangle_list, this_vertex_list, this_triangle_list);
		std::copy(this_vertex_validity.begin(), this_vertex_validity.end(), std::back_inserter(vertex_validity));
		std::copy(this_triangle_validity.begin(), this_triangle_validity.end(), std::back_inserter(triangle_validity));
	}

}

bool VolumeModeler::saveMesh(MeshVertexVector const& vertex_list, TriangleVector const& triangle_list, fs::path save_file) const
{
	return MeshUtilities::saveMesh(vertex_list, triangle_list, save_file);
}

void VolumeModeler::getTSDFBuffers(std::vector<float>& bufferDVector, std::vector<float>& bufferDWVector, std::vector<unsigned char>& bufferCVector, std::vector<float>& bufferCWVector)
{
	ModelSingleVolume* model_single = dynamic_cast<ModelSingleVolume*>(model_ptr_.get());
	if (!model_single) throw std::runtime_error ("Can't getTSDFBuffers unless MODEL_SINGLE_VOLUME");
	model_single->getBuffers(bufferDVector, bufferDWVector, bufferCVector, bufferCWVector);
}

void VolumeModeler::getBufferD(std::vector<float>& bufferDVector)
{
	ModelSingleVolume* model_single = dynamic_cast<ModelSingleVolume*>(model_ptr_.get());
	if (!model_single) throw std::runtime_error ("Can't getBufferD unless MODEL_SINGLE_VOLUME");
	model_single->getBufferD(bufferDVector);
}

void VolumeModeler::getBufferDW(std::vector<float>& bufferDWVector)
{
	ModelSingleVolume* model_single = dynamic_cast<ModelSingleVolume*>(model_ptr_.get());
	if (!model_single) throw std::runtime_error ("Can't getBufferDW unless MODEL_SINGLE_VOLUME");
	model_single->getBufferDW(bufferDWVector);
}

void VolumeModeler::getBufferC(std::vector<unsigned char>& bufferCVector)
{
	ModelSingleVolume* model_single = dynamic_cast<ModelSingleVolume*>(model_ptr_.get());
	if (!model_single) throw std::runtime_error ("Can't getBufferC unless MODEL_SINGLE_VOLUME");
	model_single->getBufferC(bufferCVector);
}

void VolumeModeler::getBufferCW(std::vector<float>& bufferCWVector)
{
	ModelSingleVolume* model_single = dynamic_cast<ModelSingleVolume*>(model_ptr_.get());
	if (!model_single) throw std::runtime_error ("Can't getBufferCW unless MODEL_SINGLE_VOLUME");
	model_single->getBufferCW(bufferCWVector);
}

void VolumeModeler::getTSDFBuffersLists(
	std::vector<boost::shared_ptr<std::vector<float> > >& bufferDVectors, 
	std::vector<boost::shared_ptr<std::vector<float> > > & bufferDWVectors, 
	std::vector<boost::shared_ptr<std::vector<unsigned char> > > & bufferCVectors, 
	std::vector<boost::shared_ptr<std::vector<float> > > & bufferCWVectors,
	std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list,
	std::vector<boost::shared_ptr<Eigen::Array3i> > & cell_counts_list)
{
	model_ptr_->getBuffersLists(bufferDVectors, bufferDWVectors, bufferCVectors, bufferCWVectors, pose_list, cell_counts_list);
}

std::string VolumeModeler::getSummaryString()
{
	std::string result;

	result += "Model summary:\n";
	result += model_ptr_->getSummary();

	return result;
}

size_t VolumeModeler::getFramesAdded() const
{
	return model_ptr_->getCameraListSize();
}

void VolumeModeler::render(Eigen::Affine3f const& camera_pose)
{
    // shouldn't need to do this here:
    //render_buffers_.resetAllBuffers();
    Eigen::Affine3f model_pose = camera_pose.inverse();
    model_ptr_->renderModel(params_.camera, model_pose, render_buffers_);

#if 0
    // debug remove: debug_mask_render
    {
        cout << "debug remove: debug_mask_render" << endl;
        cv::Mat mask = render_buffers_.getImageBufferMask().getMat();
        cv::Mat colored(mask.size(), CV_8UC3, cv::Scalar::all(0));
        for (int row = 0; row < mask.rows; ++row) {
            for (int col = 0; col < mask.cols; ++col) {
                int mask_val = mask.at<int>(row,col);
                if (mask_val > 0) {
                    colored.at<cv::Vec3b>(row,col) = cv::Vec3b(0,255,0);
                }
                else if (mask_val == -1) {
                    colored.at<cv::Vec3b>(row,col) = cv::Vec3b(0,0,255);
				}
				else if (mask_val == -2) {
                    colored.at<cv::Vec3b>(row,col) = cv::Vec3b(255,0,255);
                }
            }
        }
        cv::imshow("debug_mask_render", colored);
    }
#endif
}

// this is a bit ugly...not actually pretty
void VolumeModeler::renderAllExtraModelsPretty(Eigen::Affine3f const& camera_pose, std::vector<cv::Mat> & color_list, std::vector<cv::Mat> & normals_list)
{
    Eigen::Affine3f render_pose = camera_pose.inverse();

    color_list.clear();
    normals_list.clear();
    BOOST_FOREACH(boost::shared_ptr<ModelBase> & m, model_ptr_list_) {
        m->renderModel(params_.camera, render_pose, render_buffers_);
        cv::Mat color, normals;
        getLastRenderPretty(color, normals);
        color_list.push_back(color);
        normals_list.push_back(normals);
    }
}

void VolumeModeler::setRenderBuffers(const RenderBuffers & render_buffers)
{
    render_buffers_ = render_buffers;
}

RenderBuffers & VolumeModeler::getRenderBuffers()
{
	return render_buffers_;
}

void VolumeModeler::getLastRenderPretty(cv::Mat & render_color, cv::Mat & render_normals)
{
    render_buffers_.getRenderPretty(render_color, render_normals);
}

void VolumeModeler::getLastRender(cv::Mat & render_color, std::vector<float> & points, std::vector<float> & normals, std::vector<int> & mask)
{
	render_color = render_buffers_.getImageBufferColorImage().getMat();

	points.resize(render_color.total() * 4);
	render_buffers_.getImageBufferPoints().getBufferWrapper().readToFloatVector(points);

	normals.resize(render_color.total() * 4);
	render_buffers_.getImageBufferNormals().getBufferWrapper().readToFloatVector(normals);

	mask.resize(render_color.total());
	render_buffers_.getImageBufferMask().getBufferWrapper().readToIntVector(mask);
}

void VolumeModeler::setAlignDebugImages(bool value)
{
	alignment_ptr_->setAlignDebugImages(value);
}

void VolumeModeler::getAlignDebugImages(std::vector<cv::Mat> & image_list)
{
	alignment_ptr_->getAlignDebugImages(image_list);
}

void VolumeModeler::getPyramidDebugImages(std::vector<cv::Mat> & image_list)
{
	alignment_ptr_->getPyramidDebugImages(image_list);
}

void VolumeModeler::deallocateBuffers()
{
	model_ptr_->deallocateBuffers();
	// list?
}

void VolumeModeler::save(fs::path const& folder)
{
    boost::timer t;

	fs::path model_folder = folder / "model";

	if (!fs::exists(model_folder) && !fs::create_directories(model_folder)) {
		throw std::runtime_error("bad folder: " + model_folder.string());
	}

	model_ptr_->save(model_folder);
	// list

	if (!model_ptr_list_.empty()) {
		cout << "WARNING: save when !model_ptr_list_.empty()" << endl;
	}

	// full command line (not loaded, just for reference)
	saveFullCommandLine(folder);

	// only save features if you're using features
	if (params_.volume_modeler.use_features) {
		// previous_frame_pose
		{
			fs::path filename = folder / "previous_frame_pose.txt";
			std::fstream file(filename.string().c_str(), std::ios::out);
			file << EigenUtilities::transformToString(previous_frame_pose_) << endl;
		}

		// previous_frame_image
		{
			fs::path filename = folder / "previous_frame_image.yaml";
			cv::FileStorage fs(filename.string(), cv::FileStorage::WRITE);
			cv::write(fs, previous_frame_image_);
		}

		// previous_frame_keypoints
		if (previous_frame_keypoints_) {
			fs::path filename = folder / "previous_frame_keypoints.yaml";
			cv::FileStorage fs(filename.string(), cv::FileStorage::WRITE);
			fs << "previous_frame_keypoints" << *previous_frame_keypoints_;
		}
	}

	// mask object
	{
		mask_object_->save(folder);
	}

    cout << "VolumeModeler::save: " << folder << " took time: " << t.elapsed() << endl;
}

void VolumeModeler::load(fs::path const& folder)
{
    boost::timer t;

	fs::path model_folder = folder / "model";

	if (!fs::exists(model_folder)) {
		throw std::runtime_error("bad folder: " + model_folder.string());
	}

	reset();

	model_ptr_->load(model_folder);
	// list?

	if (params_.volume_modeler.use_features) {
		// previous_frame_pose
		{
			fs::path filename = folder / "previous_frame_pose.txt";
			if (fs::exists(filename)) {
				std::fstream file(filename.string().c_str(), std::ios::in);
				std::string s;
				std::getline(file, s);
				previous_frame_pose_ = EigenUtilities::stringToTransform(s);
			}
			else {
				cout << "Warning: use_features but file not found: " << filename << endl;
			}
		}

		// previous_frame_image
		{
			fs::path filename = folder / "previous_frame_image.yaml";
			if (fs::exists(filename)) {
				cv::FileStorage fs(filename.string(), cv::FileStorage::READ);
				cv::read(fs.getFirstTopLevelNode(), previous_frame_image_);
			}
			else {
				cout << "Warning: use_features but file not found: " << filename << endl;
			}
		}

		// previous_frame_keypoints
		{
			// start as null
			previous_frame_keypoints_.reset();
			fs::path filename = folder / "previous_frame_keypoints.yaml";
			if (fs::exists(filename)) {
				cv::FileStorage fs(filename.string(), cv::FileStorage::READ);
				if (fs.isOpened()) {
					previous_frame_keypoints_.reset(new Keypoints);
					cv::FileNode node = fs["previous_frame_keypoints"];
					cv::read(node, *previous_frame_keypoints_);
				}
				else {
					cout << "Warning: use_features but !fs.isOpened(): " << filename << endl;
				}
			}
			else {
				cout << "Warning: use_features but file not found: " << filename << endl;
			}
		}
	}

	// mask object
	{
		mask_object_->load(folder);
	}

    cout << "VolumeModeler::load: " << folder << " took time: " << t.elapsed() << endl;
}

void VolumeModeler::setMaxWeightInVolume(float new_weight)
{
	model_ptr_->setMaxWeightInVolume(new_weight);
}

void VolumeModeler::saveCameraPoses(const fs::path & filename)
{
	std::vector<CameraStructPtr> camera_list;
	model_ptr_->getAllCameras(camera_list);

	// here's the save bit
	{
		std::ofstream ofs (filename.string().c_str());
		if (!ofs.good()) throw std::runtime_error("bad filename");
		for (size_t i = 0; i < camera_list.size(); ++i) {
			ofs << camera_list[i]->ros_timestamp << " " << EigenUtilities::transformToString(camera_list[i]->pose) << endl;
		}
	}
}

void VolumeModeler::saveGraphs(const fs::path &folder)
{
    // only main model for now
    model_ptr_->saveGraphs(folder);
}

std::vector<cv::Mat> VolumeModeler::getDebugRenderImages()
{
	ModelGrid* model_grid = getModelGridOrDie();
	return model_grid->getDebugRenderImages();
}

void VolumeModeler::mergeOtherVolume(VolumeModeler & other)
{
	if (params_.volume_modeler.model_type == MODEL_SINGLE_VOLUME) {
		mergeOtherVolume(other, Eigen::Affine3f::Identity());
	}
	else if (params_.volume_modeler.model_type == MODEL_GRID) {
		ModelGrid* model_grid = dynamic_cast<ModelGrid*>(model_ptr_.get());
		ModelGrid* other_model_grid = dynamic_cast<ModelGrid*>(other.model_ptr_.get());
		if (!model_grid || !other_model_grid) throw std::runtime_error("not both MODEL_GRID");

		model_grid->mergeOtherModelGridActiveIntoActive(*other_model_grid);

		if (params_.loop_closure.loop_closure) {
			cout << "WARNING: Pose graph not updated by mergeOtherVolume!!" << endl;
		}
	}
	else if (params_.volume_modeler.model_type == MODEL_PATCH) {
		cout << "WARNING: mergeOtherVolume not implemented for MODEL_PATCH" << endl;
	}
	else {
		throw std::runtime_error("unknown model type");
	}
}

void VolumeModeler::mergeOtherVolume(VolumeModeler & other, Eigen::Affine3f const& relative_pose)
{
	if (params_.volume_modeler.model_type == MODEL_SINGLE_VOLUME) {
		if (other.params_.volume_modeler.model_type == MODEL_SINGLE_VOLUME) {
			ModelSingleVolume* model_single_volume = dynamic_cast<ModelSingleVolume*>(model_ptr_.get());
			ModelSingleVolume* other_model_single_volume = dynamic_cast<ModelSingleVolume*>(other.model_ptr_.get());
			if (!model_single_volume || !other_model_single_volume) throw std::runtime_error("not both MODEL_SINGLE_VOLUME");

			model_single_volume->mergeOther(*other_model_single_volume, relative_pose);
		}
		else if (other.params_.volume_modeler.model_type == MODEL_GRID) {
			ModelSingleVolume* model_single_volume = dynamic_cast<ModelSingleVolume*>(model_ptr_.get());
			ModelGrid* other_model_grid = dynamic_cast<ModelGrid*>(other.model_ptr_.get());
			if (!model_single_volume || !other_model_grid) throw std::runtime_error("not MODEL_SINGLE_VOLUME MODEL_GRID");

			model_single_volume->mergeOther(*other_model_grid, relative_pose);
		}
	}
	else if (params_.volume_modeler.model_type == MODEL_GRID) {
		ModelGrid* model_grid = dynamic_cast<ModelGrid*>(model_ptr_.get());
		ModelGrid* other_model_grid = dynamic_cast<ModelGrid*>(other.model_ptr_.get());
		if (!model_grid || !other_model_grid) throw std::runtime_error("not both MODEL_GRID");

		throw std::runtime_error("mergeOtherVolume with relative_pose not implemented");
	}
	else {
		throw std::runtime_error("unknown model type");
	}
}


void VolumeModeler::setSingleVolumePose(Eigen::Affine3f const& pose)
{
	if (params_.volume_modeler.model_type == MODEL_SINGLE_VOLUME) {
		ModelSingleVolume* model_single_volume = dynamic_cast<ModelSingleVolume*>(model_ptr_.get());
		if (!model_single_volume) throw std::runtime_error("not MODEL_SINGLE_VOLUME");

		model_single_volume->setPose(pose);
	}
	else {
		throw std::runtime_error("unsupported model type");
	}
}


Eigen::Affine3f VolumeModeler::getSingleVolumePose() const
{
	if (params_.volume_modeler.model_type == MODEL_SINGLE_VOLUME) {
		ModelSingleVolume* model_single_volume = dynamic_cast<ModelSingleVolume*>(model_ptr_.get());
		if (!model_single_volume) throw std::runtime_error("not MODEL_SINGLE_VOLUME");

		return model_single_volume->getPose();
	}
	else {
		throw std::runtime_error("unsupported model type");
	}
}

void VolumeModeler::debugSetSphere(float radius)
{
	if (params_.volume_modeler.model_type == MODEL_SINGLE_VOLUME) {
		ModelSingleVolume* model_single_volume = dynamic_cast<ModelSingleVolume*>(model_ptr_.get());
		if (!model_single_volume) throw std::runtime_error("not MODEL_SINGLE_VOLUME");

		model_single_volume->debugSetSphere(radius);
	}
	else if (params_.volume_modeler.model_type == MODEL_GRID) {
		ModelGrid* model_grid = dynamic_cast<ModelGrid*>(model_ptr_.get());
		if (!model_grid) throw std::runtime_error("not MODEL_GRID");

		// HERE, FOOL...TEST MULTI VOLUME COPY
		throw std::runtime_error("resize MODEL_GRID not implemented");
	}
	else {
		throw std::runtime_error("unsupported model type");
	}
}


void VolumeModeler::getNvidiaGPUMemoryUsage(int & total_mb, int & available_mb)
{
	#define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
	#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

	GLint total_mem_kb = 0;
	glGetIntegerv(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX, &total_mem_kb);

	GLint cur_avail_mem_kb = 0;
	glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &cur_avail_mem_kb);

	GLenum error = glGetError();

	if (error != GL_NO_ERROR) {
		cout << "getNvidiaGPUMemoryUsage glGetError: " << error << endl;
		if (error == GL_INVALID_ENUM) cout << "GL_INVALID_ENUM" << endl;
		if (error == GL_INVALID_VALUE) cout << "GL_INVALID_VALUE" << endl;
		if (error == GL_INVALID_OPERATION) cout << "GL_INVALID_OPERATION" << endl;
		// ther eare more at https://www.opengl.org/wiki/OpenGL_Error
	}

	total_mb = total_mem_kb >> 10;
	available_mb = cur_avail_mem_kb >> 10;
}

void VolumeModeler::setUpdateInterface(boost::shared_ptr<UpdateInterface> update_interface_ptr)
{
	this->update_interface_ = update_interface_ptr;
	model_ptr_->setUpdateInterface(update_interface_ptr);
	BOOST_FOREACH(boost::shared_ptr<ModelBase> & m, model_ptr_list_) {
		m->setUpdateInterface(update_interface_ptr);
	}
}

void VolumeModeler::refreshUpdateInterfaceForModels()
{
    model_ptr_->refreshUpdateInterface();
    BOOST_FOREACH(boost::shared_ptr<ModelBase> & m, model_ptr_list_) {
        m->refreshUpdateInterface();
    }

}

void VolumeModeler::setPickPixel(boost::shared_ptr<PickPixel> pick_pixel_ptr)
{
	this->pick_pixel_ = pick_pixel_ptr;
	model_ptr_->setPickPixel(pick_pixel_ptr);
	BOOST_FOREACH(boost::shared_ptr<ModelBase> & m, model_ptr_list_) {
		m->setPickPixel(pick_pixel_ptr);
	}
}

void VolumeModeler::setValueInSphere(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Vector3f const& center, float radius)
{
	model_ptr_->setValueInSphere(d_value, dw_value, c_value, cw_value, center, radius);
}

void VolumeModeler::setValueInBox(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Affine3f const& pose)
{
	model_ptr_->setValueInBox(d_value, dw_value, c_value, cw_value, pose);
}

ModelGrid* VolumeModeler::getModelGridOrDie()
{
	ModelGrid* model_grid = dynamic_cast<ModelGrid*>(model_ptr_.get());
	if (!model_grid) throw std::runtime_error("not MODEL_GRID");
	return model_grid;
}

std::map<std::string, cv::Mat> VolumeModeler::getDebugImages()
{
	// update with model debug images every time
	std::map<std::string, cv::Mat> const& model_debug_images = model_ptr_->getDebugImages();

	// model_ptr_list as well??

	for (std::map<std::string, cv::Mat>::const_iterator iter = model_debug_images.begin(); iter != model_debug_images.end(); ++iter) {
		debug_images_[iter->first] = iter->second;
	}
	// also masking debug images
	std::map<std::string, cv::Mat> const& mask_debug_images = mask_object_->getDebugImages();
	for (std::map<std::string, cv::Mat>::const_iterator iter = mask_debug_images.begin(); iter != mask_debug_images.end(); ++iter) {
		debug_images_[iter->first] = iter->second;
	}

	return debug_images_;
}

void VolumeModeler::debugCheckOverlap() {
	ModelGrid* model_grid = getModelGridOrDie();
	model_grid->debugCheckOverlap();
}

void VolumeModeler::debugAddVolume() {
	model_ptr_->debugAddVolume();
}

Eigen::Affine3f VolumeModeler::getLastCameraPose() {
	return model_ptr_->getLastCameraPose();
}

void VolumeModeler::getAllCameraPoses(std::vector<boost::shared_ptr<Eigen::Affine3f> > & result_pose_list)
{
    result_pose_list.clear();

    std::vector<CameraStructPtr> all_cameras;
    model_ptr_->getAllCameras(all_cameras);
    BOOST_FOREACH(CameraStructPtr & cam, all_cameras) {
        result_pose_list.push_back(boost::shared_ptr<Eigen::Affine3f>(new Eigen::Affine3f(cam->pose)));
    }
}


