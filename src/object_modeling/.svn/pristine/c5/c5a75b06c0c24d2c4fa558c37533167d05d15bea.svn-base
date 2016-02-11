#pragma once

#include "opencv_utilities.h"
#include "EigenUtilities.h"
#include "params_camera.h"
#include "ImageBuffer.h"
#include "OpenCLAllKernels.h"

namespace VignetteCalibration {

	struct Measurement {
		float radius;
		float gray_value;
		float depth;
	};

	typedef std::pair<Measurement, Measurement> Constraint;
	typedef std::vector<Constraint> ConstraintList;


	void project(boost::shared_ptr<OpenCLAllKernels> all_kernels,
		const ParamsCamera & params_camera,
		const Eigen::Affine3f & transform,
		const cv::Mat mat_depth,
		cv::Mat & result_projected_depths,
		cv::Mat & result_projected_pixels);

    void addConstraints(const ParamsCamera & params_camera,
                        const cv::Mat projected_depths, const cv::Mat projected_pixels, const cv::Mat source_gray,
                        const cv::Mat target_depths, const cv::Mat target_gray,
                        ConstraintList & constraint_list);

#if 0
	// project current points onto previous image
    void project(boost::shared_ptr<OpenCLAllKernels> all_kernels,
		const ImageBuffer & ib_points,
		const cv::Mat & color_points,
		const cv::Mat & target_depths,
		const cv::Mat & target_colors,
		const Eigen::Affine3f & pose,
		const ParamsCamera & params_camera,
		ConstraintList & result_constraint_list,
        cv::Mat & projected_gray,
		cv::Mat & projected_depths,
        cv::Mat & target_gray,
        std::map<std::string, cv::Mat> & debug_images);
#endif

	void filterConstraintsByMaxError(const ConstraintList & input, const float max_abs_error, ConstraintList & result);

	void filterConstraintsByValue(const ConstraintList & input, const float min_value, const float max_value, ConstraintList & result);

    void filterConstraintsByDepthDifference(const ConstraintList & input, const float max_depth_difference, ConstraintList & result);

	void filterConstraintsByRadiusBinning(const ConstraintList & input, const size_t radius_bin_count, const size_t radius_bin_max_per, ConstraintList & result);

	void solveLeastSquaresPolynomial3(const ConstraintList & constraint_list, Eigen::Array3f & result_model);

    void solveLeastSquaresPolynomial2(const ConstraintList & constraint_list, Eigen::Array2f & result_model);


	/////////////////////////
	// templated

	template <typename T>
	void replaceFromPixels(const cv::Mat & depths, const cv::Mat & pixels, const cv::Mat_<T> & source, cv::Mat_<T> & target)
	{
        for (int row = 0; row < target.rows; ++row) {
            for (int col = 0; col < target.cols; ++col) {
				float d = depths.at<float>(row,col);
				if (d > 0) {
					const cv::Vec2i & pixel = pixels.at<cv::Vec2i>(row,col);
                    const int pixel_row = pixel[1];
                    const int pixel_col = pixel[0];
                    target(row,col) = source(pixel_row,pixel_col);
				}
			}
		}
	}


	template <typename T>
    cv::Mat_<T> applyVignetteModelPolynomial3(const cv::Mat_<T> & input, const ParamsCamera & params_camera, const Eigen::Array3f & model)
	{
		const float radius_scale = (Eigen::Vector2f(0,0) - params_camera.center.matrix()).norm();

		// want something like this:
		//cv::Mat_<T> result(input.size(), cv::Scalar_<T>::all(0));
		// this is ok for this function
		cv::Mat_<T> result(input.size());
		for (int row = 0; row < input.rows; ++row) {
			for (int col = 0; col < input.cols; ++col) {
				float radius = (Eigen::Vector2f(col,row) - params_camera.center.matrix()).norm() / radius_scale;
				float r_2 = radius * radius;
				float r_4 = r_2 * r_2;
				float r_6 = r_2 * r_2 * r_2;
                float vignette_factor = 1 + model[0] * r_2 + model[1] * r_4 + model[2] * r_6;
				result(row,col) = input(row,col) / vignette_factor;
			}
		}

		return result;
	}

    template <typename T>
    cv::Mat_<T> applyVignetteModelPolynomial2(const cv::Mat_<T> & input, const ParamsCamera & params_camera, const Eigen::Array2f & model)
    {
        const float radius_scale = (Eigen::Vector2f(0,0) - params_camera.center.matrix()).norm();

        // want something like this:
        //cv::Mat_<T> result(input.size(), cv::Scalar_<T>::all(0));
        // this is ok for this function
        cv::Mat_<T> result(input.size());
        for (int row = 0; row < input.rows; ++row) {
            for (int col = 0; col < input.cols; ++col) {
                float radius = (Eigen::Vector2f(col,row) - params_camera.center.matrix()).norm() / radius_scale;
                float r_2 = radius * radius;
                float r_4 = r_2 * r_2;
                float vignette_factor = 1 + model[0] * r_2 + model[1] * r_4;
                result(row,col) = input(row,col) / vignette_factor;
            }
        }

        return result;
    }



} // ns
