#pragma once

// maybe too many here:
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/icp/types_icp.h>
#include <g2o/core/robust_kernel_impl.h>

// assumes PointT.x projects to KeypointT.x
template<typename PointT, typename KeypointT>
class G2OStereoProjector{
protected:

	boost::shared_ptr<g2o::VertexSCam> projection_vertex_ptr;
	void setKCamAndProjectionVertex(const Eigen::Vector2f& focal_lengths, const Eigen::Vector2f& centers, float baseline)
	{
	  g2o::VertexSCam::setKcam(focal_lengths[0], focal_lengths[1], centers[0], centers[1], baseline);
	  projection_vertex_ptr.reset(new g2o::VertexSCam);
	  Eigen::Vector3d trans(0, 0, 0);
	  Eigen::Quaterniond q;
	  q.setIdentity();
	  g2o::SE3Quat pose(q, trans);
	  projection_vertex_ptr->setEstimate(pose);
	  projection_vertex_ptr->setAll(); // set aux transforms
	}

	// params
	bool DENSE;
	bool ROBUST_KERNEL;
	double HUBER_WIDTH;
	unsigned int SBA_ITERS;

	// cam params
	Eigen::Vector2f focal_lengths_;
	Eigen::Vector2f centers_;
	float baseline_;

public:

	G2OStereoProjector(const Eigen::Vector2f& focal_lengths, const Eigen::Vector2f& centers, float baseline)
		: focal_lengths_(focal_lengths),
		centers_(centers),
		baseline_(baseline),
		DENSE(true),
		ROBUST_KERNEL(true),
		HUBER_WIDTH(1.0),
		SBA_ITERS(20)
	{
		setKCamAndProjectionVertex(focal_lengths, centers, baseline);
	}

	void fillInZ(const pcl::PointCloud<PointT>& cloud, pcl::PointCloud<KeypointT>& keypoint_cloud) const
	{
		assert(cloud.size() == keypoint_cloud.size());
		for (size_t i = 0; i < cloud.size(); i++) {
			Eigen::Vector3f source = cloud.points[i].getVector3fMap();
			Eigen::Vector3d projected;
			projection_vertex_ptr->mapPoint(projected, source.cast<double>());
			keypoint_cloud.points[i].z = (float) projected[2];
		}
	}

	void projectPointStereo(const Eigen::Vector3f& pt_3d, Eigen::Vector3f& projection) const
	{
		Eigen::Vector3d projected;
		projection_vertex_ptr->mapPoint(projected, pt_3d.cast<double>());
		projection = projected.cast<float>();
	}

	void projectPointImage(const Eigen::Vector3f& pt_3d, Eigen::Vector2f& projection) const
	{
		projection[0] = pt_3d[0] * focal_lengths_[0] / pt_3d[2] + centers_[0];
		projection[1] = pt_3d[1] * focal_lengths_[1] / pt_3d[2] + centers_[1];
	}

	const Eigen::Vector2f& getFocalLengths() const
	{
		return focal_lengths_;
	}

	const Eigen::Vector2f& getCenters() const
	{
		return centers_;
	}

	float getSquaredProjectionError(const Eigen::Vector3f& pt_3d, const KeypointT& point_for_kp) const
	{
		  Eigen::Vector3d projected;
		  projection_vertex_ptr->mapPoint(projected, pt_3d.cast<double>());
		  float result = (projected.cast<float>() - point_for_kp.getVector3fMap()).squaredNorm();

		  // what mapPoint does:
#if 0
		  // calculate stereo projection
		  void mapPoint(Vector3d &res, const Vector3d &pt3)
		  {
			Vector4d pt;
			pt.head<3>() = pt3;
			pt(3) = 1.0;
			Vector3d p1 = w2i * pt;
			Vector3d p2 = w2n * pt;
			Vector3d pb(baseline,0,0);

			double invp1 = 1.0/p1(2);
			res.head<2>() = p1.head<2>()*invp1;

			// right camera px
			p2 = Kcam*(p2-pb);
			res(2) = p2(0)/p2(2);
		  }
#endif

	  return result;
	 }

	// This is based on sba_demo.cpp from g2o
	// Assumes that g2o::VertexSCam::setKcam(focal_length[0], focal_length[1], principal_point[0], principal_point[1], baseline); has been called
	// Assumes that the KeypointT.vector3fMap() gives the correct observation, including "disparity"
	// Uses the initial value in transformation_matrix as a starting point if initialize_with_transformation_matrix
	// Note that we don't even need to set up the target camera if fix_points is true...in fact I'm going to write a function....
	// So really if you are setting fix_points=true, call the other function
	//
	// This will move CAMERA src and minimize the projection of the 3d points (initialized from target) onto both frames.
	// This is different than PCL, which moves the source POINTS so that they are closed to the target projections.
	void estimateSBATransformBetweenFrames(
			const pcl::PointCloud<PointT> &cloud_src, const pcl::PointCloud<KeypointT> &keypoints_src, const std::vector<int> &indices_src,
			const pcl::PointCloud<PointT> &cloud_tgt, const pcl::PointCloud<KeypointT> &keypoints_tgt, const std::vector<int> &indices_tgt,
			Eigen::Matrix4f &transformation_matrix, bool initialize_with_transformation_matrix = false, bool fix_points = false) const {

		assert(indices_src.size() == indices_tgt.size());
		size_t index_size = indices_src.size();
		assert(index_size >= 3);

		g2o::SparseOptimizer optimizer;
		optimizer.setVerbose(false);
		g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
		if (DENSE) {
			linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
		} else {
			linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();
		}

		g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		optimizer.setAlgorithm(solver);

		////////////////////////////////
		// add cameras
		int id_source = 0;
		int id_target = 1;
		int camera_vertex_count = 2; // for id offset

		g2o::VertexSCam * v_se3_source = new g2o::VertexSCam();
		v_se3_source->setId(id_source);
		if (initialize_with_transformation_matrix) {
			// source pose
			Eigen::Matrix3d source_r_mat = transformation_matrix.block<3,3>(0,0).cast<double>();
			Eigen::Vector3d source_t_vec = transformation_matrix.col(3).head(3).cast<double>();
			g2o::SE3Quat pose_source(source_r_mat, source_t_vec);
			v_se3_source->setEstimate(pose_source);
		}
		v_se3_source->setAll(); // set aux transforms
		optimizer.addVertex(v_se3_source);

		g2o::VertexSCam * v_se3_target = new g2o::VertexSCam();
		v_se3_target->setId(id_target);
		v_se3_target->setAll(); // set aux transforms
		v_se3_target->setFixed(true);
		optimizer.addVertex(v_se3_target);

		///////////////////////////////
		// add points
		for (size_t i = 0; i < index_size; i++) {
			// Note that we use the 3d point from target only (could average or something)
			// also, if points are fixed, we don't even need the kp_tgt (target observations)
			//const PointT& p_src = cloud_src.points[indices_src[i]];
			const KeypointT& kp_src = keypoints_src.points[indices_src[i]];
			const PointT& p_tgt = cloud_tgt.points[indices_tgt[i]];
			const KeypointT& kp_tgt = keypoints_tgt.points[indices_tgt[i]];

			//Eigen::Vector3f eigen_p_src = p_src.getVector3fMap();
			Eigen::Vector3f eigen_p_tgt = p_tgt.getVector3fMap();

			g2o::VertexPointXYZ * v_p = new g2o::VertexPointXYZ();

			v_p->setId(i + camera_vertex_count);
			v_p->setMarginalized(true);
			v_p->setFixed(fix_points);
			Eigen::Vector3f point3f = p_tgt.getVector3fMap();
			v_p->setEstimate( point3f.cast<double> () );
			optimizer.addVertex(v_p);

			Eigen::Vector3f obs_src_f = kp_src.getVector3fMap();
			Eigen::Vector3f obs_tgt_f = kp_tgt.getVector3fMap();
			Eigen::Vector3d obs_src = obs_src_f.cast<double>();
			Eigen::Vector3d obs_tgt = obs_tgt_f.cast<double>();

			// create edges for each
			g2o::Edge_XYZ_VSC * e_src = new g2o::Edge_XYZ_VSC();
			e_src->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*> (v_p);
			e_src->vertices()[1] = v_se3_source; //<g2o::OptimizableGraph::Vertex*> (optimizer.vertices().find(id_source)->second);
			e_src->setMeasurement(obs_src);
			e_src->information() = Eigen::Matrix3d::Identity();
			if (ROBUST_KERNEL) {
				g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
				rk->setDelta(HUBER_WIDTH);
				e_src->setRobustKernel(rk);
			}
			optimizer.addEdge(e_src);

			g2o::Edge_XYZ_VSC * e_tgt = new g2o::Edge_XYZ_VSC();
			e_tgt->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*> (v_p);
			e_tgt->vertices()[1] = v_se3_target; //dynamic_cast<g2o::OptimizableGraph::Vertex*> (optimizer.vertices().find(id_target)->second);
			e_tgt->setMeasurement(obs_tgt);
			e_tgt->information() = Eigen::Matrix3d::Identity();
			if (ROBUST_KERNEL) {
				g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
				rk->setDelta(HUBER_WIDTH);
				e_tgt->setRobustKernel(rk);
			}	
			optimizer.addEdge(e_tgt);
		} // for all points

		/////////////////////////////////////////
		// start solving
		optimizer.initializeOptimization();
		optimizer.setVerbose(false);
		optimizer.optimize(SBA_ITERS);

		transformation_matrix = v_se3_source->estimate().matrix().cast<float>();
	} // estimateSBATransformBetweenFrames


	/*
	 Similar to the above function, finds the transformation so that the projections of the cloud_tgt line up with keypoints_src
	 Again, note that this is opposite the PCL naming...
	*/
	void estimateSBATransformAgainstPoints(
			const pcl::PointCloud<KeypointT> &keypoints_src, const std::vector<int> &indices_src,
			const pcl::PointCloud<PointT> &cloud_tgt, const std::vector<int> &indices_tgt,
			Eigen::Matrix4f &transformation_matrix, bool initialize_with_transformation_matrix = false) const {

		assert(indices_src.size() == indices_tgt.size());
		size_t index_size = indices_src.size();
		assert(index_size >= 3);

		g2o::SparseOptimizer optimizer;
		optimizer.setVerbose(false);
		g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
		if (DENSE) {
			linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
		} else {
			linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();
		}

		g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		optimizer.setAlgorithm(solver);

		////////////////////////////////
		// add cameras
		int id_source = 0;
		int camera_vertex_count = 1; // for id offset

		g2o::VertexSCam * v_se3_source = new g2o::VertexSCam();
		v_se3_source->setId(id_source);
		if (initialize_with_transformation_matrix) {
			// source pose
			Eigen::Matrix3d source_r_mat = transformation_matrix.block<3,3>(0,0).cast<double>();
			Eigen::Vector3d source_t_vec = transformation_matrix.col(3).head(3).cast<double>();
			g2o::SE3Quat pose_source(source_r_mat, source_t_vec);
			v_se3_source->setEstimate(pose_source);
		}
		v_se3_source->setAll(); // set aux transforms
		optimizer.addVertex(v_se3_source);

		///////////////////////////////
		// add points
		for (size_t i = 0; i < index_size; i++) {
			const KeypointT& kp_src = keypoints_src.points[indices_src[i]];
			const PointT& p_tgt = cloud_tgt.points[indices_tgt[i]];

			Eigen::Vector3f eigen_p_tgt = p_tgt.getVector3fMap();

			g2o::VertexPointXYZ * v_p = new g2o::VertexPointXYZ();

			v_p->setId(i + camera_vertex_count);
			v_p->setMarginalized(true);
			v_p->setFixed(true);
			Eigen::Vector3f point3f = p_tgt.getVector3fMap();
			v_p->setEstimate( point3f.cast<double> () );
			optimizer.addVertex(v_p);

			Eigen::Vector3f obs_src_f = kp_src.getVector3fMap();
			Eigen::Vector3d obs_src = obs_src_f.cast<double>();

			// create edges for each
			g2o::Edge_XYZ_VSC * e_src = new g2o::Edge_XYZ_VSC();
			e_src->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*> (v_p);
			e_src->vertices()[1] = v_se3_source; //<g2o::OptimizableGraph::Vertex*> (optimizer.vertices().find(id_source)->second);
			e_src->setMeasurement(obs_src);
			e_src->information() = Eigen::Matrix3d::Identity();
			if (ROBUST_KERNEL) {
				g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
				rk->setDelta(HUBER_WIDTH);
				e_src->setRobustKernel(rk);
			}	
			optimizer.addEdge(e_src);
		} // for all points

		/////////////////////////////////////////
		// start solving
		optimizer.initializeOptimization();
		optimizer.setVerbose(false);
		optimizer.optimize(SBA_ITERS);

		transformation_matrix = v_se3_source->estimate().matrix().cast<float>();
	} // estimateSBATransformAgainstPoints
}; // class

