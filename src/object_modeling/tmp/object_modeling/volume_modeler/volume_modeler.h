#pragma once

#include <vector>

#include "RenderBuffers.h"
#include "OpenCLAllKernels.h"
#include "OpenCLNormals.h"

// alignment lib
#include "alignment.h"

#include "frame.h"
#include "basic.h"
#include "volume_modeler_all_params.h"
#include "model_base.h"
class ModelGrid;
#include "frustum.h"
#include "feature_matching.h"
#include "update_interface.h"
#include "pick_pixel.h" // or just class here

// mask object
#include "mask_object.h"


class VolumeModeler
{
private:
	void operator=(VolumeModeler const& other);

public:

    // static

    static ModelType modelTypeFromString(std::string model_type_string);

    // the rest:

	VolumeModeler(boost::shared_ptr<OpenCLAllKernels> all_kernels, VolumeModelerAllParams const& params);
	VolumeModeler(VolumeModeler const& other);

	void getParams(VolumeModelerAllParams & params) const;

	void reset();

	// convenience function:
	bool alignAndAddFrame(Frame& frame);

	// if first frame (means addFrame has never been called, or in other words camera list is empty)
	// will apply a centering pose to single volume instead of actually running alignment
	bool alignFrame(Frame& frame, Eigen::Affine3f& camera_pose_result);

	void addFrame(Frame& frame, Eigen::Affine3f const& camera_pose);

	bool doesModelSupportLoopClosure() const;

	bool loopClosure(Frame& frame);

	// a convenience function for generateMesh, saveMesh
	bool generateAndSaveMesh(fs::path save_file);

	// save "all" meshes to a folder
	// This is a debugging function for Peter
	bool generateAndSaveAllMeshes(fs::path save_folder);

	// generate the "main" mesh for the model
	void generateMesh(
		MeshVertexVector & vertex_list,
		TriangleVector & triangle_list);

	// generate the main mesh (including triangles along empty/unseen boundaries) and associated validity
	// valid triangles are those for which all 3 vertices are valid (but I return both redundant vectors: vertex_validity and triangle_validity)
	// post condition: vertex_validity.size() == vertex_list.size(), triangle_validity.size() == triangle_list.size()
	// works for both single and grid models
	void generateMeshAndValidity(
		MeshVertexVector & vertex_list,
		TriangleVector & triangle_list,
		std::vector<bool> & vertex_validity,
		std::vector<bool> & triangle_validity);

	// versions of generateMesh and generateMeshAndValididity which use previously extracted buffers (from getTSDFBuffersLists)
	// note that getTSDFBuffersLists works for both single and grid volumes
	// it should be safe to call these from a separate thread while doing other stuff to the volume modeler
	void generateMesh(
		const std::vector<boost::shared_ptr<std::vector<float> > > & bufferDVectors, 
		const std::vector<boost::shared_ptr<std::vector<float> > > & bufferDWVectors, 
		const std::vector<boost::shared_ptr<std::vector<unsigned char> > > & bufferCVectors, 
		const std::vector<boost::shared_ptr<std::vector<float> > > & bufferCWVectors,
		const std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list,
		const std::vector<boost::shared_ptr<Eigen::Array3i> > & cell_counts_list,
		MeshVertexVector & vertex_list,
		TriangleVector & triangle_list);



	void generateMeshAndValidity(
		const std::vector<boost::shared_ptr<std::vector<float> > > & bufferDVectors, 
		const std::vector<boost::shared_ptr<std::vector<float> > > & bufferDWVectors, 
		const std::vector<boost::shared_ptr<std::vector<unsigned char> > > & bufferCVectors, 
		const std::vector<boost::shared_ptr<std::vector<float> > > & bufferCWVectors,
		const std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list,
		const std::vector<boost::shared_ptr<Eigen::Array3i> > & cell_counts_list,
		MeshVertexVector & vertex_list,
		TriangleVector & triangle_list,
		std::vector<bool> & vertex_validity,
		std::vector<bool> & triangle_validity);

	// save a mesh (just calls OpenCLTSDF::saveMesh)
	bool saveMesh(MeshVertexVector const& vertex_list, TriangleVector const& triangle_list, fs::path save_file) const;

	// only works for single volume
	// in row major, then col, then slice, like you'd expect
	// bufferCVector is uchar4, BGRA
	void getTSDFBuffers(std::vector<float>& bufferDVector, std::vector<float>& bufferDWVector, std::vector<unsigned char>& bufferCVector, std::vector<float>& bufferCWVector);

	// only works for single volume
	// individual versions of getTSDFBuffers
	void getBufferD(std::vector<float>& bufferDVector);
	void getBufferDW(std::vector<float>& bufferDWVector);
	void getBufferC(std::vector<unsigned char>& bufferCVector);
	void getBufferCW(std::vector<float>& bufferCWVector);

	// works for all styles (single and grid)
	void getTSDFBuffersLists(
		std::vector<boost::shared_ptr<std::vector<float> > >& bufferDVectors, 
		std::vector<boost::shared_ptr<std::vector<float> > > & bufferDWVectors, 
		std::vector<boost::shared_ptr<std::vector<unsigned char> > > & bufferCVectors, 
		std::vector<boost::shared_ptr<std::vector<float> > > & bufferCWVectors,
		std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list,
		std::vector<boost::shared_ptr<Eigen::Array3i> > & cell_counts_list);
	
	void render(Eigen::Affine3f const& camera_pose);

    void setRenderBuffers(const RenderBuffers & render_buffers);
	RenderBuffers & getRenderBuffers();

    void renderAllExtraModelsPretty(Eigen::Affine3f const& camera_pose, std::vector<cv::Mat> & color_list, std::vector<cv::Mat> & normals_list);

	void getLastRenderPretty(cv::Mat & render_color, cv::Mat & render_normals);

	// the points and normals are each in groups of 4 floats in row-major order (corresponding to colors in render_color, which is BGRA uchar4)
	// in other words, to get the point at (col, row), start at index = 4*(row * rows + col)
	void getLastRender(cv::Mat & render_color, std::vector<float> & points, std::vector<float> & normals, std::vector<int> & mask);

	std::string getSummaryString(); // should be const...todo

	size_t getFramesAdded() const;

	void setAlignDebugImages(bool value);

	void getAlignDebugImages(std::vector<cv::Mat> & image_list);
	void getPyramidDebugImages(std::vector<cv::Mat> & image_list);

	// You can call this whenever you like to move memory out of the GPU memory
	// It will be automatically moved back in to GPU memory when needed
	void deallocateBuffers();

	// currently these will throw a runtime_error if you do it wrong...
	// maybe I'll switch to a bool return value at some point...
	void save(fs::path const& folder);
	void load(fs::path const& folder);

	// set the max weight in all volumes (i.e. lower confidence of volumes)
	void setMaxWeightInVolume(float new_weight);

	//////////////
	// save in Freiburg compatible format: "123451235.1235235 tx ty tz qw qx qy qz"
    void saveCameraPoses(const boost::filesystem::path &filename);

    void saveGraphs(const fs::path & folder);

	////////////////

	// debugging grid render
	std::vector<cv::Mat> getDebugRenderImages();

	// merge other volume into this one (works for single(single), single(grid), grid(grid))
	void mergeOtherVolume(VolumeModeler & other);

	// merge other volume into this one (works for single(single) and single(grid))
	void mergeOtherVolume(VolumeModeler & other, Eigen::Affine3f const& relative_pose);

	// get or set single volume pose
	void setSingleVolumePose(Eigen::Affine3f const& pose);
	Eigen::Affine3f getSingleVolumePose() const;

	// Peter only:
	void debugSetSphere(float radius);

	// uses nvidia specific OpenGL extensions to get GPU memory information (in MB)
	// THIS FUNCTION DOES NOT YET WORK!!
	void getNvidiaGPUMemoryUsage(int & total_mb, int & available_mb);

	// Currently peter only...3D viewer
	void setUpdateInterface(boost::shared_ptr<UpdateInterface> update_interface_ptr);

    // can call this externally to update non frame-specific stuff for models
    void refreshUpdateInterfaceForModels();

	// For debugging some models
	void setPickPixel(boost::shared_ptr<PickPixel> pick_pixel_ptr);

	// set d and dw (weight) for all voxels inside sphere
	// for example, to set to "known empty": d_value = 1.0, dw_value = 1.0 (dw_value is effectively the number of fake "frames" which measured this as empty)
	void setValueInSphere(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Vector3f const& center, float radius);

	// pose is applied to a (0,0,0) - (1,1,1) box
	// note that pose may include scaling
	void setValueInBox(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Affine3f const& pose);


	// get points or normals in groups of 4 floats in row-major order 
	// in other words, to get the point/normal at (col, row), start at index = 4*(row * rows + col)
	// invalid points and normals will have NaN for all 4 floats
	// not all valid points have valid normals (edge points don't)
	// will resize vectors as needed
	// will only compute points/normals once and store in frame
	void getPoints(Frame & frame, std::vector<float> & points);
	void getNormals(Frame & frame, std::vector<float> & normals);

	std::map<std::string, cv::Mat> getDebugImages();

	void debugCheckOverlap();

	void debugAddVolume();

	Eigen::Affine3f getLastCameraPose();

    void getAllCameraPoses(std::vector<boost::shared_ptr<Eigen::Affine3f> > & result_pose_list);



protected:

	boost::shared_ptr<ModelBase> allocateNewModel(ModelType model_type);
	void allocateMainModel();
	void allocateModelList();

	void ensureImageBuffers(Frame & frame);
	void ensurePointsBuffer(Frame & frame);
	void ensureNormalsBuffer(Frame & frame);

	bool isFirstFrame() const;

	void saveFullCommandLine(fs::path save_folder);

	ModelGrid* getModelGridOrDie();


	///////////////
	// members
	VolumeModelerAllParams params_;

	boost::shared_ptr<OpenCLAllKernels> all_kernels_;
	boost::shared_ptr<OpenCLNormals> opencl_normals_ptr_;

	RenderBuffers render_buffers_;

	Frustum frustum_;

	boost::shared_ptr<Alignment> alignment_ptr_;

	boost::shared_ptr<FeatureMatching> features_ptr_;

	boost::shared_ptr<UpdateInterface> update_interface_;

	std::map<std::string, cv::Mat> debug_images_;

	boost::shared_ptr<PickPixel> pick_pixel_;

	//////////////////
	// changed with state
	boost::shared_ptr<ModelBase> model_ptr_;

	std::vector<boost::shared_ptr<ModelBase> > model_ptr_list_;

	// for features, obviously...notice that initial camera pose doesn't affect these
	Eigen::Affine3f previous_frame_pose_; // to keep it consistent
	boost::shared_ptr<Keypoints> previous_frame_keypoints_;
	cv::Mat previous_frame_image_; // just for display

	// for masking (in-hand)
	boost::shared_ptr<MaskObject> mask_object_;


public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
