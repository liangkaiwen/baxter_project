#pragma once

#include <Eigen/Core>

/*
This is all broken currently
*/

template <Scalar, PointT>
class DenseColorFunctor
{
public:
	//typedef double Scalar;
	enum {
		InputsAtCompileTime = Eigen::Dynamic,
		ValuesAtCompileTime = Eigen::Dynamic
	};
	typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
	typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
	typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

	CloudT::Ptr cloud_ptr_;
	cv::Mat cloud_image_;
	Eigen::VectorXf cloud_gray_eigen_; // also cache cloud intensities?
	cv::Mat image_;
	cv::Mat image_gray_float_;
	cv::Mat gradient_x_;
	cv::Mat gradient_y_;

	/*
	* Currently assumes cloud_ptr is organized and image_for_cloud is the associated color image
	*/
	DenseColorFunctor(CloudT::Ptr cloud_ptr, cv::Mat cloud_image, cv::Mat image_to_match) :
	cloud_ptr_(cloud_ptr),
		cloud_image_(cloud_image),
		image_(image_to_match)
	{
		// todo: take existing float image
		cv::Mat image_float;
		image_.convertTo(image_float, CV_32F, 1.0/255.0);
		cv::cvtColor(image_float, image_gray_float_, CV_BGR2GRAY);
		// last argument is aperture (3 is default, -1 means sharr, 1 means 1x3 w/no smoothing)
		// NOTE: try normalizing with scale param
		float scale_for_sobel_neg1 = 1.0 / 32.0; // sharr 3,10,3
		float scale_for_sobel_3 = 1.0 / 8.0; // sobel 1,2,1
		float scale_for_sobel_1 = 1.0 / 2.0; // -1,0,1
		cv::Sobel(image_gray_float_, gradient_x_, CV_32F, 1, 0, 3, scale_for_sobel_3);
		cv::Sobel(image_gray_float_, gradient_y_, CV_32F, 0, 1, 3, scale_for_sobel_3);

		// grab the grayscale floats from the cloud for cloud_gray_
		convertColorImageToEigenIntensityVector(cloud_image_, cloud_gray_eigen_);
	}

	int values() const {
		return cloud_ptr_->size();
	}

	int operator() (const InputType &x, ValueType &fvec) const {
		cout << "functor operator(): " << endl;
		cout << "on x: " << x.transpose() << endl;

		// dup
		pcl::WarpPointRigid6D<PointT, PointT> warp_point;
		Eigen::VectorXf params = x.cast<float> ();
		warp_point.setParam (params);
		Eigen::Matrix4f x_matrix = warp_point.getTransform();
		Eigen::Affine3f x_transform(x_matrix);

		Eigen::VectorXf errors_f;

		// apply to the point cloud
		CloudT::Ptr transformed_cloud_ptr(new CloudT);
		pcl::transformPointCloud(*cloud_ptr_, *transformed_cloud_ptr, x_transform);

		// get the error
		cv::Mat debug_image;
		getProjectedCloudInterpolatedDifference(transformed_cloud_ptr, cloud_gray_eigen_, image_gray_float_, errors_f, debug_image);

		fvec = errors_f.cast<Scalar>();

		cout << "x_matrix:\n" << x_matrix << endl;
		cout << "fvec.squaredNorm(): " << fvec.squaredNorm() << endl;

		// I think this means "ok"
		return 0;
	}

	/*
	* Attempt analytical jacobian
	*
	* This is still broken
	*/
	int df (const InputType &x, JacobianType &jmat)
	{
		// first convert input to a useful transform
		// dup
		pcl::WarpPointRigid6D<PointT, PointT> warp_point;
		Eigen::VectorXf params = x.cast<float> ();
		warp_point.setParam (params);
		Eigen::Matrix4f x_matrix = warp_point.getTransform();
		Eigen::Affine3f x_transform(x_matrix);

		// apply to the point cloud
		CloudT::Ptr transformed_cloud_ptr(new CloudT);
		pcl::transformPointCloud(*cloud_ptr_, *transformed_cloud_ptr, x_transform);

		// zero out jmat
		jmat.setZero();

		// precompute I(p_x, p_y) - P_c)
		Eigen::VectorXf proj_differences;
		cv::Mat debug_image;
		getProjectedCloudInterpolatedDifference(transformed_cloud_ptr, cloud_gray_eigen_, image_gray_float_, proj_differences, debug_image);

		// precompute IGx and IGy(p_x, p_y)
		Eigen::VectorXf g_x, g_y;
		vector<int> indices_used;
		cv::Mat debug_g_x, debug_g_y;
		getProjectedCloudInterpolatedValues(transformed_cloud_ptr, gradient_x_, g_x, indices_used, debug_g_x);
		getProjectedCloudInterpolatedValues(transformed_cloud_ptr, gradient_y_, g_y, indices_used, debug_g_y);
		cv::imshow(window_2, debug_g_x);
		cv::imshow(window_3, debug_g_y);

		// loop over indices used
		for (size_t i = 0; i < indices_used.size(); i++) {
			const int& p_index = indices_used[i];
			PointT& p = transformed_cloud_ptr->points[p_index];
			// do you want 4d?
			Eigen::Vector3f p_3d = p.getVector3fMap();
			// no, you want the transformed points!

			// jacobian of 3d position with respect to params
			Eigen::MatrixXf j_3d_x = Eigen::MatrixXf::Zero(3,6);
			j_3d_x.block(0,0,3,3) = Eigen::Matrix3f::Identity();
			Eigen::Matrix3f j_quat = Eigen::Matrix3f::Zero(3,3);
			j_quat(0,1) = p_3d.z();
			j_quat(0,1) = -p_3d.z();
			j_quat(0,2) = -p_3d.y();
			j_quat(2,0) = p_3d.y();
			j_quat(1,2) = p_3d.x();
			j_quat(2,1) = -p_3d.x();
			// I believe it's x2??
			j_quat = 2 * j_quat;
			j_3d_x.block(0,3,3,3) = j_quat;

			// jacobian of projection relative to 3d position
			Eigen::Vector2f f(params.camera_focal_x, params.camera_focal_y);
			Eigen::MatrixXf j_proj_3d = Eigen::MatrixXf::Zero(2,3);
			j_proj_3d(0,0) = f(0) / p_3d.z();
			j_proj_3d(1,1) = f(1) / p_3d.z();
			j_proj_3d(0,2) = - p_3d.x() * f(0) / (p_3d.z() * p_3d.z());
			j_proj_3d(1,2) = - p_3d.y() * f(1) / (p_3d.z() * p_3d.z());

			// jacobian of error relative to pixel coordinates
			Eigen::MatrixXf j_error_proj = Eigen::MatrixXf::Zero(1,2);
			// 2(I(p_x, p_y) - P_c) * GIx(p_x, p_y)
			float de_drow = 2 * proj_differences[p_index] * g_y[p_index];
			float de_dcol = 2 * proj_differences[p_index] * g_x[p_index];
			j_error_proj(0,0) = de_dcol;
			j_error_proj(0,1) = de_drow;

			Eigen::MatrixXf jacobian_row = j_error_proj * j_proj_3d * j_3d_x;
			// debug:
			//			cout << "j_error_proj: \n" << j_error_proj  << endl;
			//			cout << "j_proj_3d: \n" << j_proj_3d  << endl;
			//			cout << "j_3d_x: \n" << j_3d_x  << endl;
			//			cout << "jacobian_row: \n" << jacobian_row << endl;

			jmat.row(p_index) = jacobian_row.row(0).cast<double>();

			// debug
			//cout << "jmat.row(p_index): " << jmat.row(p_index) << endl;
		}
		return 0;
	}
};