#pragma once

#include "cll.h"
#include "RenderBuffers.h"
#include "OpenCLAllKernels.h"

#include "volume_modeler_all_params.h"

#include "basic.h"
#include "frame.h"
#include "update_interface.h"
#include "camera_struct.h"

#include "alignment.h"

#include "ros_timestamp.h"

#include "MeshTypes.h"

// shouldn't actually need this for model base...when I moved meshes out it broke a bunch of stuff though
#include "OpenCLTSDF.h"

#include "pick_pixel.h"

class ModelBase
{
private:
	ModelBase& operator=(ModelBase const& other);

public:
	ModelBase(boost::shared_ptr<OpenCLAllKernels> all_kernels, VolumeModelerAllParams const& params, boost::shared_ptr<Alignment> alignment_ptr, RenderBuffers const& render_buffers);

	ModelBase(ModelBase const& other);

	virtual void reset();

	virtual ModelBase* clone() = 0;

	virtual bool isEmpty() const;

	virtual size_t getCameraListSize() const;

    virtual void prepareForRenderCurrent();

	virtual void renderModel(
		const ParamsCamera & params_camera,
        const Eigen::Affine3f & model_pose,
		RenderBuffers & render_buffers) = 0;

	virtual void updateModel(
		Frame & frame,
		const Eigen::Affine3f & model_pose);

	virtual void generateMesh(
		MeshVertexVector & vertex_list,
		TriangleVector & triangle_list) = 0;

	virtual void generateMeshAndValidity(
		MeshVertexVector & vertex_list,
		TriangleVector & triangle_list,
		std::vector<bool> & vertex_validity,
		std::vector<bool> & triangle_validity) = 0;

	virtual void generateAllMeshes(std::vector<std::pair<std::string, MeshPtr> > & names_and_meshes);

	virtual void getBuffersLists(
		std::vector<boost::shared_ptr<std::vector<float> > >& bufferDVectors, 
		std::vector<boost::shared_ptr<std::vector<float> > > & bufferDWVectors, 
		std::vector<boost::shared_ptr<std::vector<unsigned char> > > & bufferCVectors, 
		std::vector<boost::shared_ptr<std::vector<float> > > & bufferCWVectors,
		std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list,
        std::vector<boost::shared_ptr<Eigen::Array3i> > & cell_counts_list);

	virtual std::string getSummary() {return std::string();} // default to nothing

	virtual void deallocateBuffers() = 0;

	virtual void save(fs::path const& folder);

	virtual void load(fs::path const& folder);

    virtual void setMaxWeightInVolume(float new_weight);

    virtual void setValueInSphere(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Vector3f const& center, float radius);

    virtual void setValueInBox(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Affine3f const& pose);

    virtual void getNonzeroVolumePointCloud(MeshVertexVector & vertex_list);

	virtual void setUpdateInterface(boost::shared_ptr<UpdateInterface> update_interface_ptr);

	virtual void setPickPixel(boost::shared_ptr<PickPixel> pick_pixel_ptr);

	virtual void setDebugStringPrefix(const std::string & s);

	virtual void refreshUpdateInterface();

	virtual std::map<std::string, cv::Mat> const& getDebugImages() const;

	virtual void debugAddVolume();

	virtual Eigen::Affine3f getLastCameraPose() const;

	virtual bool loopClosure(Frame& frame);

	virtual void getAllCameras(std::vector<CameraStructPtr> & cameras_list);

    // returns false if no such camera
    virtual bool getCameraPose(size_t i, Eigen::Affine3f & result_pose);

	// utility for single volume models
	virtual Eigen::Affine3f getSingleVolumePoseForFirstFrame(Frame& frame);

    virtual void saveGraphs(const fs::path & folder);

protected:
	// members
	boost::shared_ptr<OpenCLAllKernels> all_kernels_;

	VolumeModelerAllParams const& params_;

	boost::shared_ptr<UpdateInterface> update_interface_;

	boost::shared_ptr<PickPixel> pick_pixel_;

	std::string debug_string_prefix_;

	std::map<std::string, cv::Mat> debug_images_;

	boost::shared_ptr<Alignment> alignment_ptr_;

	RenderBuffers render_buffers_;

	// need to save:
	std::vector<CameraStructPtr> camera_list_;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
