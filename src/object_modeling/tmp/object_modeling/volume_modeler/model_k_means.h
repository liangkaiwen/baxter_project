#pragma once

#include "model_base.h"

#include "pick_pixel.h"
#include "trackbar_window.h"

#include "VolumeBuffer.h"
#include <vector>



class ModelKMeans : public ModelBase
{
public:
    ModelKMeans(boost::shared_ptr<OpenCLAllKernels> all_kernels, VolumeModelerAllParams const& params, boost::shared_ptr<Alignment> alignment_ptr, RenderBuffers const& render_buffers);
	//ModelKMeans(ModelKMeans const& other);

	// clone broken until you do deep copies
    virtual ModelKMeans* clone();

	virtual void reset();

	// inherited

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

	virtual void generateAllMeshes(std::vector<std::pair<std::string, MeshPtr> > & names_and_meshes);

    virtual void deallocateBuffers();

	virtual void save(fs::path const& folder);

	virtual void load(fs::path const& folder);

	virtual void refreshUpdateInterface();

    virtual void getBoundingLines(MeshVertexVector & vertex_list);


	// unique to this:




protected:
	// functions

    bool getMeansAndCountsForVoxel(const Eigen::Array3i & voxel, std::vector<float> & result_means, std::vector<float> & result_counts);

    bool getMeansAndCountsForPoint(const Eigen::Vector3f & world_point, std::vector<float> & result_means, std::vector<float> & result_counts);

    void computeMinAbsVolume(float minimum_weight_fraction, VolumeBuffer & combined_mean, VolumeBuffer & combined_weight);

	std::vector<MeshPtr> getPartialMeshes();

    std::vector<MeshPtr> getValidPartialMeshesByNormal(const std::vector<MeshPtr> & mesh_list, boost::shared_ptr<std::vector<MeshPtr> > result_also_invalid = boost::shared_ptr<std::vector<MeshPtr> >());

    std::vector<MeshPtr> getValidPartialMeshesByEmptyViolation(const std::vector<MeshPtr> & mesh_list, boost::shared_ptr<std::vector<MeshPtr> > result_also_invalid = boost::shared_ptr<std::vector<MeshPtr> >());

	// members:
    
	std::vector<boost::shared_ptr<VolumeBuffer> > buffer_mean_list_;
	std::vector<boost::shared_ptr<VolumeBuffer> > buffer_count_list_;

    // for checking the stupid way of doing normals:
    std::vector<boost::shared_ptr<VolumeBuffer> > buffer_normal_list_;

	static std::vector<Eigen::Vector3f> fixed_normal_list_;
	static std::vector<Eigen::Array4ub> fixed_color_list_;
    static std::vector<Eigen::AngleAxisf> fixed_normal_cameras_;

	Eigen::Affine3f volume_pose_;

	// duplicated in models
	Eigen::Array2i last_pick_pixel_;
	Eigen::Vector3f last_pick_pixel_world_point_;
    Eigen::Vector3f last_pick_pixel_camera_point_;

    // here?
    TrackbarWindow trackbar_window_;

    // debug remove?
    std::vector<bool> debug_compatible_last_frame_;
	std::vector<bool> debug_best_last_frame_;


public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
