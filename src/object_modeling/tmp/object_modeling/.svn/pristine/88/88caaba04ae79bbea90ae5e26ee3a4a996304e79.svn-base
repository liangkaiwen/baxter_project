#pragma once


#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "MeshTypes.h"

class UpdateInterface
{
public:
	typedef std::vector<Eigen::Affine3f, Eigen::aligned_allocator<Eigen::Affine3f> > PoseListT;
	typedef boost::shared_ptr<PoseListT> PoseListPtrT;
	typedef std::vector<std::pair<int,int> > EdgeListT;
	typedef boost::shared_ptr<EdgeListT> EdgeListPtrT;

	virtual void updateMesh(const std::string & name, MeshPtr mesh_ptr) {};
	virtual void updateAlphaMesh(const std::string & name, MeshPtr mesh_ptr) {};
	virtual void updateLines(const std::string & name, MeshVertexVectorPtr vertices_ptr) {};
	virtual void updatePointCloud(const std::string & name, MeshVertexVectorPtr vertices_ptr) {};
	virtual void updateCameraList(const std::string & name, PoseListPtrT pose_list_ptr) {};
	virtual void updateGraph(const std::string & name, PoseListPtrT vertices, EdgeListPtrT edges) {};
	virtual void updateBipartiteGraph(const std::string & name, PoseListPtrT vertices_first, PoseListPtrT vertices_second, EdgeListPtrT edges) {};

    virtual void updateScale(const std::string & name, float scale) {};
	virtual void updateColor(const std::string & name, const Eigen::Array4ub & color) {};

	// if you want to show a pose, use updateCameraList
	// this sets virtual camera pose
	virtual void updateViewPose(Eigen::Affine3f const& pose) {};
};
