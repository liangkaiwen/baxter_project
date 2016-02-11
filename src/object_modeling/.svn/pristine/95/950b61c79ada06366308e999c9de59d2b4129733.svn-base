#pragma once

#include "model_grid.h"

class ModelMovingVolumeGrid : public ModelGrid 
{
public:
	ModelMovingVolumeGrid(boost::shared_ptr<OpenCLAllKernels> all_kernels, VolumeModelerAllParams const& params, boost::shared_ptr<Alignment> alignment_ptr, RenderBuffers const& render_buffers);
	ModelMovingVolumeGrid(ModelMovingVolumeGrid const& other);

	ModelMovingVolumeGrid* clone();

	// takes care of base classes too
	virtual void reset();

    virtual void prepareForRenderCurrent();

    virtual void renderModel(
            const ParamsCamera & params_camera,
            const Eigen::Affine3f & model_pose,
            RenderBuffers & render_buffers);

	virtual void updateModel(
		Frame & frame,
		const Eigen::Affine3f & model_pose);

	virtual void save(fs::path const& folder);

	virtual void load(fs::path const& folder);

	virtual void refreshUpdateInterface();

	// hacking this in for debugging
	virtual void generateMesh(
		MeshVertexVector & vertex_list,
		TriangleVector & triangle_list);

	// todo?
#if 0
	virtual void generateMeshAndValidity(
		MeshVertexVector & vertex_list, 
		TriangleVector & triangle_list, 
		std::vector<bool> & vertex_validity, 
		std::vector<bool> & triangle_validity);
#endif

	// mostly uses ModelGrid, but must merge afterwards if success
	virtual bool loopClosure(Frame& frame);

protected:

	// from constructor (just this class)
	void resetThisClass();

	void resetMovingVolume();

	// returns the index of the added grid, or -1 if no grid was added
	int appendGridCellFromMovingVolume(OpenCLTSDF & tsdf, Eigen::Affine3f const& tsdf_pose, Eigen::Array3i const& voxel);

	Eigen::Affine3f getBlockTransform(Eigen::Affine3f const& tsdf_pose, Eigen::Array3i const& voxel);

	// we need to hook up additional edges to constrain the single volume and recent keyframes
	virtual void createG2OPoseGraphKeyframes(
		G2OPoseGraph & pose_graph,
		std::map<int, int> & keyframe_to_vertex_map,
		std::map<int, int> & volume_to_vertex_map);

	void mergeVolumesIntoMovingVolume();

	//////////////////////////
	// members

	bool prepare_for_render_loop_closure_;

	// need to save:

    boost::shared_ptr<OpenCLTSDF> moving_tsdf_ptr_;
    Eigen::Affine3f moving_tsdf_pose_; // model_pose * tsdf_pose is ultimate model pose

	std::map<int, std::vector<int> > keyframe_to_grid_creation_;

	// another idea for how to link keyframes and spit-out blocks
	typedef boost::tuple<int,int,int> BlockIndexT;
	std::map<BlockIndexT, std::set<int> > block_index_to_keyframe_;
	Eigen::Array3i moving_volume_block_offset_;

};
