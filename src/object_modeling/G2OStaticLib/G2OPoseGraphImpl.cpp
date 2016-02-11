
#include "G2OPoseGraphImpl.h"

using std::cout;
using std::cerr;
using std::endl;
using std::exception;


G2OPoseGraphImpl::G2OPoseGraphImpl()
	: vertex_id_counter(0),
	optimizer()
{
#if 1
	initOptimizer();
#else
	PoseGraphLinearSolverCSparse* linear_solver = new PoseGraphLinearSolverCSparse();
	linear_solver->setBlockOrdering(false); // I don't understand the difference or defaults
	PoseGraphBlockSolver* block_solver = new PoseGraphBlockSolver(linear_solver);
	g2o::OptimizationAlgorithm* solver = new g2o::OptimizationAlgorithmLevenberg(block_solver);
	optimizer.setAlgorithm(solver);
#endif
}

G2OPoseGraphImpl::G2OPoseGraphImpl(G2OPoseGraphImpl const& other)
	: vertex_id_counter(other.vertex_id_counter),
	optimizer()
{
	initOptimizer();
	// now copy the graph over...through save and load?
	std::stringstream ss;
	other.optimizer.save(ss);
	optimizer.load(ss);
}

void G2OPoseGraphImpl::initOptimizer()
{
	PoseGraphLinearSolverCSparse* linear_solver = new PoseGraphLinearSolverCSparse();
	linear_solver->setBlockOrdering(false); // I don't understand the difference or defaults
	PoseGraphBlockSolver* block_solver = new PoseGraphBlockSolver(linear_solver);
	g2o::OptimizationAlgorithm* solver = new g2o::OptimizationAlgorithmLevenberg(block_solver);
	optimizer.setAlgorithm(solver);
}

int G2OPoseGraphImpl::addVertex(const Eigen::Isometry3d& pose, bool fixed)
{
	g2o::VertexSE3* v = new g2o::VertexSE3();
	v->setEstimate(pose);
	v->setFixed(fixed);
	v->setId(vertex_id_counter++);
	bool success = optimizer.addVertex(v); // check?
	return v->id();
}

// information assumes (x,y,z,qx,qy,qz)
bool G2OPoseGraphImpl::addEdge(size_t v1, size_t v2, Eigen::Isometry3d const& relative_pose, InformationMatrixT const& information_matrix)
{
	// Used to do this (debug access violation in hash table?)
	// These should throw an exception if v1 or v2 is invalid (don't do that!)
	//g2o::VertexSE3* v1_ptr = dynamic_cast<g2o::VertexSE3*>( optimizer.vertices().find(v1)->second);
	//g2o::VertexSE3* v2_ptr = dynamic_cast<g2o::VertexSE3*>( optimizer.vertices().find(v2)->second);
	g2o::VertexSE3* v1_ptr = dynamic_cast<g2o::VertexSE3*>( optimizer.vertex(v1));
	g2o::VertexSE3* v2_ptr = dynamic_cast<g2o::VertexSE3*>( optimizer.vertex(v2));
	// but this "edges()" still then causes problems...what the hell?

	// don't add an edge between vertices that already have an edge?  Probably wrong...
	for (g2o::SparseOptimizer::EdgeSet::iterator iter = v1_ptr->edges().begin(); iter != v1_ptr->edges().end(); ++iter) {
		if ((*iter)->vertex(1) == v2_ptr) return false;
	}
	g2o::EdgeSE3* e = new g2o::EdgeSE3();
	e->vertices()[0] = v1_ptr;
	e->vertices()[1] = v2_ptr;
	e->setMeasurement(relative_pose);
	//e->setInformation(g2o::EdgeSE3::InformationType::Identity()); // todo: take arbitrary information matrix
	e->setInformation(information_matrix);
	optimizer.addEdge(e); // This returns false and does nothing if IT'S THE EXACT SAME EDGE POINTER
	return true;
}

bool G2OPoseGraphImpl::addEdge(size_t v1, size_t v2, Eigen::Isometry3d const& relative_pose, double information_scale)
{
	InformationMatrixT information_matrix = information_scale * InformationMatrixT::Identity();
	return addEdge(v1,v2,relative_pose,information_matrix);
}

bool G2OPoseGraphImpl::addEdge(size_t v1, size_t v2, Eigen::Isometry3d const& relative_pose)
{
	return addEdge(v1,v2,relative_pose,InformationMatrixT::Identity());
}

void G2OPoseGraphImpl::optimize(int iterations)
{
	optimizer.initializeOptimization();
#if 1
	optimizer.optimize(iterations);
#else
	for (int i = 0; i < iterations; ++i) {
		cout << "running optimize " << i << endl;
		optimizer.optimize(1);
		float chi2 = optimizer.activeRobustChi2();
		cout << "optimizer.activeRobustChi2() " << chi2 << endl;
	}
#endif
}

void G2OPoseGraphImpl::getVertexPose(int v, Eigen::Isometry3d& pose)
{
	pose = dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(v)->second)->estimate();
}

Eigen::Isometry3d G2OPoseGraphImpl::getVertexPose(int v)
{
	Eigen::Isometry3d result;
	getVertexPose(v, result);
	return result;
}

bool G2OPoseGraphImpl::hasVertex(size_t v)
{
	return (optimizer.vertices().find(v) != optimizer.vertices().end());
}

size_t G2OPoseGraphImpl::getVertexCount()
{
	return optimizer.vertices().size();
}

size_t G2OPoseGraphImpl::getEdgeCount()
{
	return optimizer.edges().size();
}

void G2OPoseGraphImpl::getEdges(std::vector<std::pair<size_t, size_t> >& result)
{
	result.clear();
	for (g2o::SparseOptimizer::EdgeSet::iterator iter = optimizer.edges().begin(); iter != optimizer.edges().end(); ++iter) {
		result.push_back(std::make_pair((*iter)->vertices()[0]->id(), (*iter)->vertices()[1]->id()));
	}
}

void G2OPoseGraphImpl::setVerbose(bool value)
{
	optimizer.setVerbose(value);
}

void G2OPoseGraphImpl::setVertexFixed(int v, bool value)
{
	optimizer.vertex(v)->setFixed(value);
}

bool G2OPoseGraphImpl::save(fs::path const& folder)
{
	// create folder if needed
	if (!fs::exists(folder) && !fs::create_directories(folder)) return false;

	{
		fs::path filename = folder / "vertex_id_counter.txt";
		std::ofstream file(filename.string().c_str());
		file << vertex_id_counter;
	}

	{
		fs::path filename = folder / "optimizer.txt";
		return optimizer.save(filename.string().c_str());
	}
}

bool G2OPoseGraphImpl::load(fs::path const& folder)
{
	{
		fs::path filename = folder / "vertex_id_counter.txt";
		std::ifstream file(filename.string().c_str());
		file >> vertex_id_counter;
	}

	{
		fs::path filename = folder / "optimizer.txt";
		return optimizer.load(filename.string().c_str());
	}
}
