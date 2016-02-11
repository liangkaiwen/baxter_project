#pragma once

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/icp/types_icp.h>
#include <g2o/core/robust_kernel_impl.h>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

class G2OPoseGraphImpl
{
private:
	G2OPoseGraphImpl& operator=(G2OPoseGraphImpl const& other);

public:
	typedef Eigen::Matrix<double, 6, 6> InformationMatrixT;

	G2OPoseGraphImpl();
	G2OPoseGraphImpl(G2OPoseGraphImpl const& other);

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
	typedef g2o::BlockSolver< g2o::BlockSolverTraits<-1, -1> > PoseGraphBlockSolver;
	typedef g2o::LinearSolverCSparse<PoseGraphBlockSolver::PoseMatrixType> PoseGraphLinearSolverCSparse;

	// methods
	void initOptimizer();

	// members
	g2o::SparseOptimizer optimizer;
	size_t vertex_id_counter;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

