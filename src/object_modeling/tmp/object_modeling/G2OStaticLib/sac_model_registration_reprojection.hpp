#ifndef PCL_SAMPLE_CONSENSUS_IMPL_SAC_MODEL_REGISTRATION_REPROJECTION_H_
#define PCL_SAMPLE_CONSENSUS_IMPL_SAC_MODEL_REGISTRATION_REPROJECTION_H_

#include "sac_model_registration_reprojection.h"

#include <exception>

using std::cout;
using std::endl;

//////////////////////////////////////////////////////////////////////////
template <typename PointT, typename KeypointT> bool
pcl_peter::SampleConsensusModelRegistrationReprojection<PointT, KeypointT>::isSampleGood (const std::vector<int> &samples) const
{
  return ((input_->points[samples[1]].getArray4fMap () - input_->points[samples[0]].getArray4fMap ()).matrix ().squaredNorm () > sample_dist_thresh_ && 
		  (input_->points[samples[2]].getArray4fMap () - input_->points[samples[0]].getArray4fMap ()).matrix ().squaredNorm () > sample_dist_thresh_ && 
		  (input_->points[samples[2]].getArray4fMap () - input_->points[samples[1]].getArray4fMap ()).matrix ().squaredNorm () > sample_dist_thresh_);
}

//////////////////////////////////////////////////////////////////////////
template <typename PointT, typename KeypointT> bool
pcl_peter::SampleConsensusModelRegistrationReprojection<PointT, KeypointT>::computeModelCoefficients (const std::vector<int> &samples, Eigen::VectorXf &model_coefficients)
{
  if (!target_)
  {
	PCL_ERROR ("[pcl_peter::SampleConsensusModelRegistrationReprojection::computeModelCoefficients] No target dataset given!\n");
	return (false);
  }
  // Need 3 samples
  if (samples.size () != 3)
	return (false);

  std::vector<int> indices_tgt (3);
  for (int i = 0; i < 3; ++i)
	indices_tgt[i] = correspondences_[samples[i]];

  estimateRigidTransformationSVD (*input_, samples, *target_, indices_tgt, model_coefficients);
  return (true);
}

//////////////////////////////////////////////////////////////////////////
template <typename PointT, typename KeypointT> void
pcl_peter::SampleConsensusModelRegistrationReprojection<PointT, KeypointT>::getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances)
{
  if (indices_->size () != indices_tgt_->size ())
  {
	PCL_ERROR ("[pcl_peter::SampleConsensusModelRegistrationReprojection::getDistancesToModel] Number of source indices (%lu) differs than number of target indices (%lu)!\n", (unsigned long)indices_->size (), (unsigned long)indices_tgt_->size ());
	distances.clear ();
	return;
  }
  if (!target_)
  {
	PCL_ERROR ("[pcl_peter::SampleConsensusModelRegistrationReprojection::getDistanceToModel] No target dataset given!\n");
	return;
  }
  // Check if the model is valid given the user constraints
  if (!isModelValid (model_coefficients))
  {
	distances.clear ();
	return;
  }
  distances.resize (indices_->size ());

  // Get the 4x4 transformation
  Eigen::Matrix4f transform;
  transform.row (0) = model_coefficients.segment<4>(0);
  transform.row (1) = model_coefficients.segment<4>(4);
  transform.row (2) = model_coefficients.segment<4>(8);
  transform.row (3) = model_coefficients.segment<4>(12);

  for (size_t i = 0; i < indices_->size (); ++i)
  {
		//original:
#if 0
	Eigen::Vector4f pt_src = input_->points[(*indices_)[i]].getVector4fMap ();
	pt_src[3] = 1;
	Eigen::Vector4f pt_tgt = target_->points[(*indices_tgt_)[i]].getVector4fMap ();
	pt_tgt[3] = 1;

	Eigen::Vector4f p_tr = transform * pt_src;

	// Calculate the distance from the transformed point to its correspondence
	// need to compute the real norm here to keep MSAC and friends general
	distances[i] = (p_tr - pt_tgt).norm ();
#endif
	Eigen::Vector4f pt_src = input_->points[(*indices_)[i]].getVector4fMap ();
	pt_src[3] = 1;
	Eigen::Vector4f p_tr  = transform * pt_src;
	const KeypointT& point_target = target_keypoints_->points[(*indices_tgt_)[i]];
	float distance_squared = projector_.getSquaredProjectionError(p_tr.head(3), point_target);
	distances[i] = sqrt(distance_squared);
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename PointT, typename KeypointT> void
pcl_peter::SampleConsensusModelRegistrationReprojection<PointT, KeypointT>::selectWithinDistance (const Eigen::VectorXf &model_coefficients, const double threshold, std::vector<int> &inliers)
{
  if (indices_->size () != indices_tgt_->size ())
  {
	PCL_ERROR ("[pcl_peter::SampleConsensusModelRegistrationReprojection::selectWithinDistance] Number of source indices (%lu) differs than number of target indices (%lu)!\n", (unsigned long)indices_->size (), (unsigned long)indices_tgt_->size ());
	inliers.clear ();
	return;
  }
  if (!target_)
  {
	PCL_ERROR ("[pcl_peter::SampleConsensusModelRegistrationReprojection::selectWithinDistance] No target dataset given!\n");
	return;
  }

  double thresh = threshold * threshold;

  // Check if the model is valid given the user constraints
  if (!isModelValid (model_coefficients))
  {
	inliers.clear ();
	return;
  }
  
  inliers.resize (indices_->size ());

  Eigen::Matrix4f transform;
  transform.row (0) = model_coefficients.segment<4>(0);
  transform.row (1) = model_coefficients.segment<4>(4);
  transform.row (2) = model_coefficients.segment<4>(8);
  transform.row (3) = model_coefficients.segment<4>(12);

  int nr_p = 0; 
  for (size_t i = 0; i < indices_->size (); ++i)
  {
	// original:
#if 0
	Eigen::Vector4f pt_src = input_->points[(*indices_)[i]].getVector4fMap ();
	pt_src[3] = 1;
	Eigen::Vector4f pt_tgt = target_->points[(*indices_tgt_)[i]].getVector4fMap ();
	pt_tgt[3] = 1;

	Eigen::Vector4f p_tr  = transform * pt_src;

	// Calculate the distance from the transformed point to its correspondence
	if ((p_tr - pt_tgt).squaredNorm () < thresh)
	  inliers[nr_p++] = (*indices_)[i];
#endif
	Eigen::Vector4f pt_src = input_->points[(*indices_)[i]].getVector4fMap ();
	pt_src[3] = 1;
	Eigen::Vector4f p_tr  = transform * pt_src;
	const KeypointT& point_target = target_keypoints_->points[(*indices_tgt_)[i]];
	if (projector_.getSquaredProjectionError(p_tr.head(3), point_target) < thresh)
		inliers[nr_p++] = (*indices_)[i];
  }
  inliers.resize (nr_p);
} 

//////////////////////////////////////////////////////////////////////////
template <typename PointT, typename KeypointT> int
pcl_peter::SampleConsensusModelRegistrationReprojection<PointT, KeypointT>::countWithinDistance (
	const Eigen::VectorXf &model_coefficients, const double threshold)
{
  if (indices_->size () != indices_tgt_->size ())
  {
	PCL_ERROR ("[pcl_peter::SampleConsensusModelRegistrationReprojection::countWithinDistance] Number of source indices (%lu) differs than number of target indices (%lu)!\n", (unsigned long)indices_->size (), (unsigned long)indices_tgt_->size ());
	return (0);
  }
  if (!target_)
  {
	PCL_ERROR ("[pcl_peter::SampleConsensusModelRegistrationReprojection::countWithinDistance] No target dataset given!\n");
	return (0);
  }

  double thresh = threshold * threshold;

  // Check if the model is valid given the user constraints
  if (!isModelValid (model_coefficients))
	return (0);
  
  Eigen::Matrix4f transform;
  transform.row (0) = model_coefficients.segment<4>(0);
  transform.row (1) = model_coefficients.segment<4>(4);
  transform.row (2) = model_coefficients.segment<4>(8);
  transform.row (3) = model_coefficients.segment<4>(12);

  int nr_p = 0; 
  for (size_t i = 0; i < indices_->size (); ++i)
  {
		// original
#if 0
	Eigen::Vector4f pt_src = input_->points[(*indices_)[i]].getVector4fMap ();
	pt_src[3] = 1;
	Eigen::Vector4f pt_tgt = target_->points[(*indices_tgt_)[i]].getVector4fMap ();
	pt_tgt[3] = 1;

	Eigen::Vector4f p_tr  = transform * pt_src;

	// Calculate the distance from the transformed point to its correspondence
	if ((p_tr - pt_tgt).squaredNorm () < thresh)
	  nr_p++;
#endif
	Eigen::Vector4f pt_src = input_->points[(*indices_)[i]].getVector4fMap ();
	pt_src[3] = 1;
	Eigen::Vector4f p_tr  = transform * pt_src;
	const KeypointT& point_target = target_keypoints_->points[(*indices_tgt_)[i]];
	float distance_squared = projector_.getSquaredProjectionError(p_tr.head(3), point_target);
	if (distance_squared < thresh)
		nr_p++;
  }
  return (nr_p);
} 

//////////////////////////////////////////////////////////////////////////
template <typename PointT, typename KeypointT> void
pcl_peter::SampleConsensusModelRegistrationReprojection<PointT, KeypointT>::optimizeModelCoefficients (const std::vector<int> &inliers, const Eigen::VectorXf &model_coefficients, Eigen::VectorXf &optimized_coefficients)
{
  if (indices_->size () != indices_tgt_->size ())
  {
	PCL_ERROR ("[pcl_peter::SampleConsensusModelRegistrationReprojection::optimizeModelCoefficients] Number of source indices (%lu) differs than number of target indices (%lu)!\n", (unsigned long)indices_->size (), (unsigned long)indices_tgt_->size ());
	optimized_coefficients = model_coefficients;
	return;
  }

  // Check if the model is valid given the user constraints
  if (!isModelValid (model_coefficients) || !target_)
  {
	optimized_coefficients = model_coefficients;
	return;
  }

  std::vector<int> indices_src (inliers.size ());
  std::vector<int> indices_tgt (inliers.size ());
  for (size_t i = 0; i < inliers.size (); ++i)
  {
	// NOTE: not tested!
	indices_src[i] = (*indices_)[inliers[i]];
	indices_tgt[i] = (*indices_tgt_)[inliers[i]];
  }

#if 0
  // original
  estimateRigidTransformationSVD (*input_, indices_src, *target_, indices_tgt, optimized_coefficients);
#elif 0
  typename pcl_peter::TransformationEstimationReprojection<PointT, PointT>::Ptr lm_estimator(
		  new pcl_peter::TransformationEstimationReprojection<PointT, PointT>(projection_matrix_));

  Eigen::Matrix4f transform;
  transform.setIdentity();
  lm_estimator->estimateRigidTransformation (*input_, indices_src, *target_, indices_tgt, transform);

  // from matrix to vector
  optimized_coefficients.resize (16);
  optimized_coefficients.segment<4> (0) = transform.row(0);
  optimized_coefficients.segment<4> (4) = transform.row(1);
  optimized_coefficients.segment<4> (8) = transform.row(2);
  optimized_coefficients.segment<4> (12) = transform.row(3);
#else
  Eigen::Matrix4f transform;
  transform.setIdentity();
  // start with model_coefficients
  bool start_at_model_coefficients = true;
  if (start_at_model_coefficients) {
	  transform.row(0) = model_coefficients.segment<4>(0);
	  transform.row(1) = model_coefficients.segment<4>(4);
	  transform.row(2) = model_coefficients.segment<4>(8);
	  transform.row(3) = model_coefficients.segment<4>(12);
  }
  if (fix_point_positions_in_sba_) {
	  // Here we need to invert the transform, because SBA moves the camera against points, not points themselves....
	  Eigen::Matrix4f transform_inverse = transform.inverse();
	  projector_.estimateSBATransformAgainstPoints(*target_keypoints_, indices_tgt, *input_, indices_src, transform_inverse, start_at_model_coefficients);
	  transform = transform_inverse.inverse();
  }
  else {
	  //throw std::exception("Gotta fix this ordering peter!");
	  cout << "in broken path" << endl;
	  exit(1);
	  projector_.estimateSBATransformBetweenFrames(*input_, *input_keypoints_, indices_src, *target_, *target_keypoints_, indices_tgt, transform, start_at_model_coefficients, fix_point_positions_in_sba_);
  }
  // from matrix to vector
  optimized_coefficients.resize (16);
  optimized_coefficients.segment<4> (0) = transform.row(0);
  optimized_coefficients.segment<4> (4) = transform.row(1);
  optimized_coefficients.segment<4> (8) = transform.row(2);
  optimized_coefficients.segment<4> (12) = transform.row(3);

#endif



}

//////////////////////////////////////////////////////////////////////////
template <typename PointT, typename KeypointT> void
pcl_peter::SampleConsensusModelRegistrationReprojection<PointT, KeypointT>::estimateRigidTransformationSVD (
	const pcl::PointCloud<PointT> &cloud_src,
	const std::vector<int> &indices_src, 
	const pcl::PointCloud<PointT> &cloud_tgt,
	const std::vector<int> &indices_tgt, 
	Eigen::VectorXf &transform)
{
  transform.resize (16);
  Eigen::Vector4f centroid_src, centroid_tgt;
  // Estimate the centroids of source, target
  compute3DCentroid (cloud_src, indices_src, centroid_src);
  compute3DCentroid (cloud_tgt, indices_tgt, centroid_tgt);

  // Subtract the centroids from source, target
  Eigen::MatrixXf cloud_src_demean;
  demeanPointCloud (cloud_src, indices_src, centroid_src, cloud_src_demean);

  Eigen::MatrixXf cloud_tgt_demean;
  demeanPointCloud (cloud_tgt, indices_tgt, centroid_tgt, cloud_tgt_demean);

  // Assemble the correlation matrix H = source * target'
  Eigen::Matrix3f H = (cloud_src_demean * cloud_tgt_demean.transpose ()).topLeftCorner<3, 3>();

  // Compute the Singular Value Decomposition
  Eigen::JacobiSVD<Eigen::Matrix3f> svd (H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3f u = svd.matrixU ();
  Eigen::Matrix3f v = svd.matrixV ();

  // Compute R = V * U'
  if (u.determinant () * v.determinant () < 0)
  {
	for (int x = 0; x < 3; ++x)
	  v (x, 2) *= -1;
  }

  Eigen::Matrix3f R = v * u.transpose ();

  // Return the correct transformation
  transform.segment<3> (0) = R.row (0); transform[12]  = 0;
  transform.segment<3> (4) = R.row (1); transform[13]  = 0;
  transform.segment<3> (8) = R.row (2); transform[14] = 0;

  Eigen::Vector3f t = centroid_tgt.head<3> () - R * centroid_src.head<3> ();
  transform[3] = t[0]; transform[7] = t[1]; transform[11] = t[2]; transform[15] = 1.0;
}

//#define PCL_INSTANTIATE_SampleConsensusModelRegistrationReprojection(T) template class PCL_EXPORTS pcl_peter::SampleConsensusModelRegistrationReprojection<T>;

#endif    // PCL_SAMPLE_CONSENSUS_IMPL_SAC_MODEL_REGISTRATION_REPROJECTION_H_

