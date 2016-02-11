
#include <iostream>
using std::cout;
using std::endl;


// g2o stuff directly instead?
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/icp/types_icp.h>
#include <g2o/core/robust_kernel_impl.h>


int main(int argc, char* argv[])
{

    /////////////////////////////////
#if 0
    {
        cout << "Test 1" << endl;
        G2OPoseGraph pose_graph;
        Eigen::Isometry3d vertex_pose_g2o = EigenUtilities::getIsometry3d(Eigen::Affine3f::Identity());
        int vertex_id = pose_graph.addVertex(vertex_pose_g2o, false);
    }
#endif

#if 1
    // is it something about my "impl"?  NO NO!  IT'S G2O'S FAULT!
    {

                cout << "Test 2" << endl;
        // from header (..impl.h"
        typedef g2o::BlockSolver< g2o::BlockSolverTraits<-1, -1> > PoseGraphBlockSolver;
        typedef g2o::LinearSolverCSparse<PoseGraphBlockSolver::PoseMatrixType> PoseGraphLinearSolverCSparse;
            g2o::SparseOptimizer optimizer;

            // from "initOPtimizer
        PoseGraphLinearSolverCSparse* linear_solver = new PoseGraphLinearSolverCSparse();
        linear_solver->setBlockOrdering(false); // I don't understand the difference or defaults
        PoseGraphBlockSolver* block_solver = new PoseGraphBlockSolver(linear_solver);
        g2o::OptimizationAlgorithm* solver = new g2o::OptimizationAlgorithmLevenberg(block_solver);
        optimizer.setAlgorithm(solver);

        // from addVertex
        g2o::VertexSE3* v = new g2o::VertexSE3();
        v->setId(0);
        bool success = optimizer.addVertex(v);

    }
#endif

    cout << "End of tests" << endl;

    return 0;
}
