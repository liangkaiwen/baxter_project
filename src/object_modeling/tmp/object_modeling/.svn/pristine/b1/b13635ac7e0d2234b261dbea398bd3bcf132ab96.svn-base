
#include "G2OPoseGraph.h"
#include "G2OPoseGraphImpl.h"


G2OPoseGraph::G2OPoseGraph()
	: impl(new G2OPoseGraphImpl())
{
}

G2OPoseGraph::G2OPoseGraph(G2OPoseGraph const& other)
	: impl(new G2OPoseGraphImpl(*other.impl))
{
}

int G2OPoseGraph::addVertex(const Eigen::Isometry3d& pose, bool fixed)
{
	return impl->addVertex(pose, fixed);
}

// information assumes (x,y,z,qx,qy,qz)
bool G2OPoseGraph::addEdge(size_t v1, size_t v2, Eigen::Isometry3d const& relative_pose, InformationMatrixT const& information_matrix)
{
	return impl->addEdge(v1, v2, relative_pose, information_matrix);
}

bool G2OPoseGraph::addEdge(size_t v1, size_t v2, Eigen::Isometry3d const& relative_pose, double information_scale)
{
	return impl->addEdge(v1, v2, relative_pose, information_scale);
}

bool G2OPoseGraph::addEdge(size_t v1, size_t v2, Eigen::Isometry3d const& relative_pose)
{
	return impl->addEdge(v1, v2, relative_pose);
}

void G2OPoseGraph::optimize(int iterations)
{
	impl->optimize(iterations);
}

void G2OPoseGraph::getVertexPose(int v, Eigen::Isometry3d& pose)
{
	impl->getVertexPose(v, pose);
}

Eigen::Isometry3d G2OPoseGraph::getVertexPose(int v)
{
	return impl->getVertexPose(v);
}

bool G2OPoseGraph::hasVertex(size_t v)
{
	return impl->hasVertex(v);
}

size_t G2OPoseGraph::getVertexCount()
{
	return impl->getVertexCount();
}

size_t G2OPoseGraph::getEdgeCount()
{
	return impl->getEdgeCount();
}

void G2OPoseGraph::getEdges(std::vector<std::pair<size_t, size_t> >& result)
{
	impl->getEdges(result);
}

void G2OPoseGraph::setVerbose(bool value)
{
	impl->setVerbose(value);
}

void G2OPoseGraph::setVertexFixed(int v, bool value)
{
	impl->setVertexFixed(v, value);
}

bool G2OPoseGraph::save(fs::path const& folder)
{
	return impl->save(folder);
}

bool G2OPoseGraph::load(fs::path const& folder)
{
	return impl->load(folder);
}
