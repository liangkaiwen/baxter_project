#include "feature_matching.h"

#include "image_to_cloud.h"

FeatureMatching::FeatureMatching(ParamsCamera const& params_camera, ParamsFeatures const& params_features)
	: params_camera(params_camera),
	params_features(params_features),
	verbose(false)
{
	if (params_features.feature_type == FEATURE_TYPE_FAST) {
		// FAST / BRIEF
		detector_ptr = cv::Ptr<cv::FeatureDetector>(new cv::FastFeatureDetector(params_features.fast_threshold));
		if (params_features.fast_pyramid_adapter) {
			detector_ptr = cv::Ptr<cv::FeatureDetector>(new cv::PyramidAdaptedFeatureDetector(detector_ptr));
		}
		if (params_features.fast_grid_adapter) {
			detector_ptr = cv::Ptr<cv::FeatureDetector>(new cv::GridAdaptedFeatureDetector(detector_ptr, params_features.max_features));
		}
		int briefBytes = 32; // 16, 32, 64 are only ones (32 is default)
		extractor_ptr = cv::Ptr<cv::DescriptorExtractor>(new cv::BriefDescriptorExtractor(briefBytes));
		matcher_ptr = cv::Ptr<cv::DescriptorMatcher>(new cv::BFMatcher(cv::NORM_L1));
	}
	else if (params_features.feature_type == FEATURE_TYPE_SURF) {
		// SURF
		int minHessian = 300; // 400 default 
		int nOctaves = 2; // 3 default
		int nOctaveLayers = 4; // 4 default
		bool upright = false; // false default
		detector_ptr = cv::Ptr<cv::FeatureDetector>(new cv::SurfFeatureDetector(minHessian, nOctaves, nOctaveLayers, upright));
		extractor_ptr = cv::Ptr<cv::DescriptorExtractor>(new cv::SurfDescriptorExtractor());
		matcher_ptr = cv::Ptr<cv::DescriptorMatcher>(new cv::BFMatcher(cv::NORM_L2));
	}
	else if (params_features.feature_type == FEATURE_TYPE_ORB) {
		// there are probably other parameters
		detector_ptr = cv::Ptr<cv::FeatureDetector>(new cv::OrbFeatureDetector(params_features.max_features));
		extractor_ptr = cv::Ptr<cv::DescriptorExtractor>(new cv::OrbDescriptorExtractor()); // patch size??
		matcher_ptr = cv::Ptr<cv::DescriptorMatcher>(new cv::BFMatcher(cv::NORM_L2));
	}
	else {
		throw std::runtime_error("unknown feature type");
	}
}

void FeatureMatching::computeFeatures(const Frame & frame, Keypoints & keypoints)
{
	keypoints.keypoints.clear();
	keypoints.descriptors = cv::Mat();

	cv::Mat mask = frame.mat_depth > 0;
	detector_ptr->detect(frame.mat_color_bgra, keypoints.keypoints, mask);
	extractor_ptr->compute(frame.mat_color_bgra, keypoints.keypoints, keypoints.descriptors); // may change keypoints

	// go through and compute a 3D point for each keypoint.  
	// If 3D point is invalid (possible due to complicated detectors), remove the keypoint?
	// NO, instead leave as nan point
	// std::vector<bool> keep_keypoint(keypoints.keypoints.size(), true);
	keypoints.points = Eigen::Matrix4Xf(4, keypoints.keypoints.size());
	for (size_t i = 0; i < keypoints.keypoints.size(); ++i) {
		cv::KeyPoint & k = keypoints.keypoints[i];
		// could interpolate in depth image, but easiest is to round
		int depth_col = (int) (k.pt.x + 0.5);
		int depth_row = (int) (k.pt.y + 0.5);
		float d = frame.mat_depth.at<float>(depth_row, depth_col);
		if (d > 0) {
			// can use fractional or rounded pixel here...
			Eigen::Array2f pixel(k.pt.x, k.pt.y);
			keypoints.points.col(i) = depthToPoint(params_camera, pixel, d);
		}
		else {
			keypoints.points.col(i).fill(std::numeric_limits<float>::quiet_NaN());
		}
	}
}

void FeatureMatching::matchDescriptors(const Keypoints & keypoints_source, const Keypoints & keypoints_target, std::vector< cv::DMatch > & matches)
{
	matches.clear();
	if (keypoints_source.keypoints.empty() || keypoints_target.keypoints.empty()) return; // avoid some exception?

	if (params_features.use_ratio_test) {
		std::vector<std::vector<cv::DMatch> > knn_matches;
		const int k = 2;
		matcher_ptr->knnMatch(keypoints_source.descriptors, keypoints_target.descriptors, knn_matches, k);
		for (size_t i = 0; i < knn_matches.size(); i++) {
			if (knn_matches[i][0].distance / knn_matches[i][1].distance < params_features.ratio_test_ratio) {
				matches.push_back(knn_matches[i][0]);
			}
		}
	}
	else {
		matcher_ptr->match(keypoints_source.descriptors, keypoints_target.descriptors, matches);
	}
}

bool FeatureMatching::ransacOpenCV(const Keypoints & keypoints_source, const Keypoints & keypoints_target, const std::vector<cv::DMatch> & matches, Eigen::Affine3f & pose, std::vector<cv::DMatch> & inliers)
{
	inliers.clear();

	// try using the built-in opencv version
	// will move target 3D points to line up with source projections
	std::vector<cv::Point3f> target_3d_points;
	std::vector<cv::Point2f> source_2d_points;
	for (size_t i = 0; i < matches.size(); ++i) {
		Eigen::Vector4f p = keypoints_target.points.col(matches[i].trainIdx);
		target_3d_points.push_back(cv::Point3f(p.x(), p.y(), p.z()));
		source_2d_points.push_back(keypoints_source.keypoints[matches[i].queryIdx].pt);

		if (p.x() != p.x()) {
			cout << "nan in points" << endl;
		}
	}
	cv::Mat camera_matrix(3,3,CV_32F,cv::Scalar(0));
	camera_matrix.at<float>(0,0) = params_camera.focal.x();
	camera_matrix.at<float>(1,1) = params_camera.focal.y();
	camera_matrix.at<float>(0,2) = params_camera.center.x();
	camera_matrix.at<float>(1,2) = params_camera.center.y();
	camera_matrix.at<float>(2,2) = 1;

	cv::Mat rvec = cv::Mat(3,1,CV_32F,cv::Scalar(0));
	cv::Mat tvec = cv::Mat(3,1,CV_32F,cv::Scalar(0));
	bool use_guess = true;
	int iterations = 500;
	float reprojection_error = 3.0; // 8.0???
	int min_inliers = 30; // 100?
	std::vector<int> inlier_indices;
	int flags = CV_ITERATIVE;

	// some isContinuous() on 308 in solvePNP throws on blankish frames?
	if ((int)matches.size() < min_inliers) return false;

	try {
		cv::solvePnPRansac(
			target_3d_points, 
			source_2d_points, 
			camera_matrix, cv::noArray(), 
			rvec, tvec, use_guess,
			iterations, reprojection_error, min_inliers,
			inlier_indices, flags);
	}
	catch (cv::Exception e) {
		cout << "cv::Exception in solvePnPRansac: " << e.what() << endl;
		return false;
	}

	cout << "matches size: " << matches.size() << endl;
	cout << "inlier_indices size: " << inlier_indices.size() << endl;

	// set my inlier matches
	for (size_t i = 0; i < inlier_indices.size(); ++i) {
		inliers.push_back(matches[inlier_indices[i]]);
	}

	// put the result in eigen transform
	pose.setIdentity();
	cv::Mat mat_rot;
	cv::Rodrigues(rvec, mat_rot);

	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			pose.matrix()(r,c) = (float)mat_rot.at<double>(r,c);
		}
		pose.matrix()(r,3) = (float)tvec.at<double>(r);
	}


	return (int)inliers.size() > min_inliers;
}


bool FeatureMatching::ransac(const Keypoints & keypoints_source, const Keypoints & keypoints_target, const std::vector<cv::DMatch> & matches, Eigen::Affine3f & pose, std::vector<cv::DMatch> & inliers)
{
	pose.setIdentity();
	inliers.clear();

	int N = 1000;
	const int min_inliers = 15;
	const float prob = 0.999f;
	float inlier_distance = 3.0;
	float inlier_dist_squared = inlier_distance * inlier_distance;
	if ((int)matches.size() < min_inliers) return false;

	typedef boost::mt19937 RNGType;
	RNGType rng; // or  RNGType rng( time(0) );   
	boost::uniform_int<> matches_uniform( 0, matches.size() - 1 );
	boost::variate_generator< RNGType, boost::uniform_int<> > match_sample(rng, matches_uniform);

	std::vector<int> best_inliers;
	Eigen::Affine3f best_transform;

	for (int iteration = 0; iteration < N; ++iteration) {
		int sample_set[3];
		// first try, assume selecting duplicates will simply lead to failure
		for (int i = 0; i < 3; ++i) {
			sample_set[i] = match_sample();
		}

		// set up two 3x3 eigen matrices to find the transform which moves SOURCE to line up with TARGET
		Eigen::Matrix3f source_points;
		Eigen::Matrix3f target_points;
		for (int i = 0; i < 3; ++i) {
			source_points.col(i) = keypoints_source.points.col(matches[sample_set[i]].queryIdx).head<3>();
			target_points.col(i) = keypoints_target.points.col(matches[sample_set[i]].trainIdx).head<3>();
		}
		Eigen::Affine3f sample_t (Eigen::umeyama(source_points, target_points, false));
		
		// move all source points by sample transform, count inliers with target projections
		// note this is ALL points, not just matched points
		Eigen::Matrix4Xf source_transformed = sample_t * keypoints_source.points;

		std::vector<int> this_inliers;
		for (int i = 0; i < (int)matches.size(); ++i) {
			Eigen::Array2f projection = pointToPixel(params_camera, source_transformed.col(matches[i].queryIdx));
			cv::Point2f const& pt_cv = keypoints_target.keypoints[matches[i].trainIdx].pt;
			Eigen::Array2f pt_eigen;
			pt_eigen.x() = pt_cv.x;
			pt_eigen.y() = pt_cv.y;
			float squared_reprojection_error = (projection.matrix() - pt_eigen.matrix()).squaredNorm();
			if (squared_reprojection_error < inlier_dist_squared) {
				this_inliers.push_back(i);
			}
		}
		//cout << "this_inliers: " << this_inliers.size() << endl;

		if (this_inliers.size() > best_inliers.size()) {
			best_inliers = this_inliers;
			best_transform = sample_t;

			// update termination adaptively
			float epsilon = 1 - (float)this_inliers.size()/(float)matches.size();
			float new_n = log(1-prob)/log(1-(1-epsilon)*(1-epsilon)*(1-epsilon));
			//cout << "new_n: " << new_n << endl;
			if (new_n > 0 && new_n < (float)N) {
				N = (int)(new_n+1);
			}
		}
	}

	if (verbose) {
		cout << "Final N: " << N << endl;
		cout << "best_inliers: " << best_inliers.size() << endl;
	}

	for (int i = 0; i < (int)best_inliers.size(); ++i) {
		inliers.push_back(matches[best_inliers[i]]);
	}

	// don't bother optimizing if too few inliers
	if (inliers.size() < min_inliers) return false;

	////////////////////
	// optimize result

#if 0
	// try opencv for this part
	cv::Mat camera_matrix(3,3,CV_32F,cv::Scalar(0));
	camera_matrix.at<float>(0,0) = params_camera.focal.x();
	camera_matrix.at<float>(1,1) = params_camera.focal.y();
	camera_matrix.at<float>(0,2) = params_camera.center.x();
	camera_matrix.at<float>(1,2) = params_camera.center.y();
	camera_matrix.at<float>(2,2) = 1;

	cv::Mat rvec = cv::Mat(3,1,CV_32F,cv::Scalar(0));
	cv::Mat tvec = cv::Mat(3,1,CV_32F,cv::Scalar(0));
	///////
	// init rvec and tvec with best transform?
	Eigen::Matrix3f rmat_eigen = best_transform.matrix().topLeftCorner<3,3>();
	cv::Mat rmat;
	cv::eigen2cv(rmat_eigen, rmat);
	cv::Rodrigues(rmat, rvec);
	Eigen::Vector3f tvec_eigen = best_transform.translation();
	cv::eigen2cv(tvec_eigen, tvec);
	///////
	bool use_guess = true;
	int iterations = 500;
	float reprojection_error = 3.0; // 8.0???
	int flags = CV_ITERATIVE;

	std::vector<cv::Point3f> source_3d_points;
	std::vector<cv::Point2f> target_2d_points;
	for (size_t i = 0; i < inliers.size(); ++i) {
		Eigen::Vector4f p = keypoints_source.points.col(inliers[i].queryIdx);
		source_3d_points.push_back(cv::Point3f(p.x(), p.y(), p.z()));
		target_2d_points.push_back(keypoints_target.keypoints[inliers[i].trainIdx].pt);

		if (p.x() != p.x()) {
			cout << "-------WARNING: nan in points" << endl;
		}
	}

	try {
		cv::solvePnP(
			source_3d_points,
			target_2d_points,
			camera_matrix, cv::noArray(),
			rvec, tvec,
			use_guess, flags);
	}
	catch (cv::Exception e) {
		cout << "cv::Exception in solvePnP: " << e.what() << endl;
		return false;
	}

	// put the result in eigen transform
	pose.setIdentity();
	cv::Mat mat_rot;
	cv::Rodrigues(rvec, mat_rot);
	cv::cv2eigen(mat_rot, rmat_eigen);
	cv::cv2eigen(tvec, tvec_eigen);
	pose.matrix().topLeftCorner<3,3>() = rmat_eigen;
	pose.matrix().col(3).head<3>() = tvec_eigen;

	cout << "pose.matrix():\n" << pose.matrix() << endl;

#else
	pose = best_transform; // or don't bother optimizing :)
#endif


	return true;
}


void FeatureMatching::addFeaturesForFrameIfNeeded(Frame & frame)
{
	if (!frame.keypoints) {
		frame.keypoints.reset(new Keypoints);
		computeFeatures(frame, *frame.keypoints);
	}
}