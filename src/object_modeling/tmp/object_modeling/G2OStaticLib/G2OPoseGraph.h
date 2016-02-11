#pragma once

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

class G2OPoseGraphImpl;

class G2OPoseGraph
{
private:
	G2OPoseGraph& operator=(G2OPoseGraph const& other);

public:
	typedef Eigen::Matrix<double, 6, 6> InformationMatrixT;

	G2OPoseGraph();
	G2OPoseGraph(G2OPoseGraph const& other);

	int addVertex(const Eigen::Isometry3d& pose, bool fixed);
	/*
	If there is already ANY edge from v1 to v2, this does nothing and returns false.
	That's probably wrong, like most of the code here
	*/
	// information assumes (x,y,z,qx,qy,qz)
	bool addEdge(size_t v1, size_t v2, Eigen::Isometry3d const& relative_pose, InformationMatrixT const& information_matrix);
	bool addEdge(size_t v1, size_t v2, Eigen::Isometry3d const& relative_pose, double information_scale);
	bool addEdge(size_t v1, size_t v2, Eigen::Isometry3d const& relative_pose);
	void optimize(int iterations);
	void getVertexPose(int v, Eigen::Isometry3d& pose);
	Eigen::Isometry3d getVertexPose(int v);
	bool hasVertex(size_t v);
	size_t getVertexCount();
	size_t getEdgeCount();
	void getEdges(std::vector<std::pair<size_t, size_t> >& result);
	void setVerbose(bool value);
	void setVertexFixed(int v, bool value);
	bool save(fs::path const& folder);
	bool load(fs::path const& folder);

protected:
	boost::shared_ptr<G2OPoseGraphImpl> impl;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

