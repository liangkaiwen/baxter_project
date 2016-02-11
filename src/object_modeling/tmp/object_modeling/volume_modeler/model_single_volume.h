#pragma once

#include "model_base.h"
class ModelGrid;

class ModelSingleVolume : public ModelBase
{
public:
	ModelSingleVolume(boost::shared_ptr<OpenCLAllKernels> all_kernels, VolumeModelerAllParams const& params, boost::shared_ptr<Alignment> alignment_ptr, RenderBuffers const& render_buffers);
	ModelSingleVolume(ModelSingleVolume const& other);

	virtual ModelSingleVolume* clone();

	virtual void reset();

    virtual void renderModel(
            const ParamsCamera & params_camera,
            const Eigen::Affine3f & model_pose,
            RenderBuffers & render_buffers);

	virtual void updateModel(
		Frame & frame,
		const Eigen::Affine3f & model_pose);

	virtual void generateMesh(
		MeshVertexVector & vertex_list,
		TriangleVector & triangle_list);

	virtual void generateMeshAndValidity(
		MeshVertexVector & vertex_list, 
		TriangleVector & triangle_list, 
		std::vector<bool> & vertex_validity, 
		std::vector<bool> & triangle_validity);

	virtual void getBuffersLists(
		std::vector<boost::shared_ptr<std::vector<float> > >& bufferDVectors, 
		std::vector<boost::shared_ptr<std::vector<float> > > & bufferDWVectors, 
		std::vector<boost::shared_ptr<std::vector<unsigned char> > > & bufferCVectors, 
		std::vector<boost::shared_ptr<std::vector<float> > > & bufferCWVectors,
		std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list,
		std::vector<boost::shared_ptr<Eigen::Array3i> > & cell_counts_list);

	virtual void deallocateBuffers();

	virtual void save(fs::path const& folder);

	virtual void load(fs::path const& folder);

	virtual void setMaxWeightInVolume(float new_weight);

	virtual void setValueInSphere(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Vector3f const& center, float radius);

	virtual void setValueInBox(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Affine3f const& pose);

	virtual void getNonzeroVolumePointCloud(MeshVertexVector & vertex_list);

	virtual void debugAddVolume();

	virtual void refreshUpdateInterface();

	//////////
	// just for single volume
	void getBuffers(std::vector<float>& bufferDVector, std::vector<float>& bufferDWVector, std::vector<unsigned char>& bufferCVector, std::vector<float>& bufferCWVector);
	void getBufferD(std::vector<float>& bufferDVector);
	void getBufferDW(std::vector<float>& bufferDWVector);
	void getBufferC(std::vector<unsigned char>& bufferCVector);
	void getBufferCW(std::vector<float>& bufferCWVector);

	Eigen::Affine3f getPose() const;
	void setPose(Eigen::Affine3f const& pose);

	void mergeOther(ModelSingleVolume & other, Eigen::Affine3f const& relative_pose);

	void mergeOther(ModelGrid & other, Eigen::Affine3f const& relative_pose);

	void debugSetSphere(float radius);

	void getBoundingMesh(MeshVertexVector & vertex_list, TriangleVector & triangle_list);

	void getBoundingLines(MeshVertexVector & vertex_list);

protected:
	boost::shared_ptr<OpenCLTSDF> opencl_tsdf_ptr_;

	Eigen::Affine3f tsdf_pose_; // model_pose * tsdf_pose is ultimate model pose

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
