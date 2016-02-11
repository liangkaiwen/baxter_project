#include "vignette_calibration.h"

#include "KernelDepthImageToPoints.h"
#include "KernelTransformPoints.h"
#include "util.h" // hmm

#include <boost/foreach.hpp>

namespace VignetteCalibration {

// gives in the TARGET image space the mapping to depth in the TARGET frame and mapping back to source pixels
void project(boost::shared_ptr<OpenCLAllKernels> all_kernels,
             const ParamsCamera & params_camera,
             const Eigen::Affine3f & transform,
             const cv::Mat mat_depth,
             cv::Mat & result_projected_depths,
             cv::Mat & result_projected_pixels)
{
    KernelDepthImageToPoints _KernelDepthImageToPoints(*all_kernels);
    KernelTransformPoints _KernelTransformPoints(*all_kernels);

    ImageBuffer ib_depth(all_kernels->getCL());
    ib_depth.setMat(mat_depth);

    ImageBuffer ib_points(all_kernels->getCL());
    _KernelDepthImageToPoints.runKernel(ib_depth, ib_points, params_camera.focal, params_camera.center);

    ImageBuffer ib_points_transformed(all_kernels->getCL());
    _KernelTransformPoints.runKernel(ib_points, ib_points_transformed, transform);

    cv::Mat points_transformed = ib_points_transformed.getMat();
    projectPixels(params_camera, points_transformed, result_projected_depths, result_projected_pixels);
}

void addConstraints(const ParamsCamera & params_camera,
                    const cv::Mat projected_depths, const cv::Mat projected_pixels, const cv::Mat source_gray,
                    const cv::Mat target_depths, const cv::Mat target_gray,
                    ConstraintList & constraint_list)
{
    const int rows = projected_pixels.rows;
    const int cols = projected_pixels.cols;
    const float radius_scale = (Eigen::Vector2f(0,0) - params_camera.center.matrix()).norm();

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            // only check projected depths here (can filter by target depths later if you want)
            float projected_depth = projected_depths.at<float>(row,col);
            if (projected_depth > 0) {
                cv::Vec2i source_pixel = projected_pixels.at<cv::Vec2i>(row,col);
                int source_row = source_pixel[1];
                int source_col = source_pixel[0];

                Measurement projected_measurement;
                projected_measurement.radius = (Eigen::Vector2f(source_col, source_row) - params_camera.center.matrix()).norm() / radius_scale;
                projected_measurement.gray_value = source_gray.at<float>(source_row,source_col);
                projected_measurement.depth = projected_depth;

                Measurement target_measurement;
                target_measurement.radius = (Eigen::Vector2f(col, row) - params_camera.center.matrix()).norm() / radius_scale;
                target_measurement.gray_value = target_gray.at<float>(row,col);
                target_measurement.depth = target_depths.at<float>(row,col);


                // order shouldn't really matter for constraint
                // but does matter if you involve differing exposures / white balance
                constraint_list.push_back(std::make_pair(projected_measurement, target_measurement));
            }
        }
    }

}

// both long and wrong
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
             std::map<std::string, cv::Mat> & debug_images)
{
    KernelTransformPoints _KernelTransformPoints(*all_kernels);
    ImageBuffer ib_points_transformed(all_kernels->getCL());
    _KernelTransformPoints.runKernel(ib_points, ib_points_transformed, pose);

    cv::Mat points_transformed = ib_points_transformed.getMat();
    cv::Mat projected_colors;
    cv::Mat projected_pixels;
    projectColorsAndDepths(params_camera, points_transformed, color_points, projected_depths, projected_colors);
    projectPixels(params_camera, points_transformed, projected_depths, projected_pixels);
    cv::Mat projected_mask = projected_depths > 0;

    // get float grayscale (results and used here)

    cv::cvtColor(projected_colors, projected_gray, CV_BGRA2GRAY);
    projected_gray.convertTo(projected_gray, CV_32F, 1./255);

    cv::cvtColor(target_colors, target_gray, CV_BGRA2GRAY);
    target_gray.convertTo(target_gray, CV_32F, 1./255);


    // color sum
    if (false) {
        cv::Mat projected_plus_target = projected_colors * 0.5 + target_colors * 0.5;
        debug_images["projected_plus_target"] = projected_plus_target;
    }

    // gray sum
    if (false) {
        cv::Mat projected_plus_target_gray = projected_gray * 0.5 + target_gray * 0.5;
        debug_images["projected_plus_target_gray"] = projected_plus_target_gray;
    }

    // replace
    {
        cv::Mat project_replace = target_gray.clone();
        projected_gray.copyTo(project_replace, projected_mask);
        debug_images["project_replace"] = project_replace;
    }

    // overlay error
    {
        cv::Mat project_error = projected_gray - target_gray + 0.5;
        cv::Mat project_error_masked(project_error.size(), CV_32F, cv::Scalar::all(0.5));
        project_error.copyTo(project_error_masked, projected_mask);
        debug_images["project_error_masked"] = project_error_masked;
    }

    // actually compute correspondences / constraints
    const float radius_scale = (Eigen::Vector2f(0,0) - params_camera.center.matrix()).norm();
    for (int row = 0; row < color_points.rows; ++row) {
        for (int col = 0; col < color_points.cols; ++col) {
            // also check target depth??
            if (projected_depths.at<float>(row,col) > 0) {
                Measurement projected_measurement;
                projected_measurement.radius = (Eigen::Vector2f(col,row) - params_camera.center.matrix()).norm() / radius_scale;
                projected_measurement.gray_value = projected_gray.at<float>(row,col);
                projected_measurement.depth = projected_depths.at<float>(row,col);
                cv::Vec2i pixel = projected_pixels.at<cv::Vec2i>(row,col);
                Measurement target_measurement;
                target_measurement.radius = (Eigen::Vector2f(pixel[0],pixel[1]) - params_camera.center.matrix()).norm() / radius_scale;
                target_measurement.gray_value = target_gray.at<float>(pixel[1],pixel[0]);
                target_measurement.depth = target_depths.at<float>(pixel[1],pixel[0]);

                // order shouldn't really matter for constraint
                // but does matter if you involve differing exposures / white balance
                result_constraint_list.push_back(std::make_pair(projected_measurement, target_measurement));
            }
        }
    }
}
#endif


void filterConstraintsByMaxError(const ConstraintList & input, const float max_abs_error, ConstraintList & result)
{
    result.clear();
    BOOST_FOREACH(const VignetteCalibration::ConstraintList::value_type & p, input) {
        if (max_abs_error < 0 || fabs(p.first.gray_value - p.second.gray_value) <= max_abs_error) {
            result.push_back(p);
        }
    }
}

void filterConstraintsByValue(const ConstraintList & input, const float min_value, const float max_value, ConstraintList & result)
{
    result.clear();
    BOOST_FOREACH(const VignetteCalibration::ConstraintList::value_type & p, input) {
        if (p.first.gray_value >= min_value && p.first.gray_value <= max_value && p.second.gray_value >= min_value && p.second.gray_value <= max_value) {
            result.push_back(p);
        }
    }
}

void filterConstraintsByDepthDifference(const ConstraintList & input, const float max_depth_difference, ConstraintList & result)
{
    result.clear();
    BOOST_FOREACH(const VignetteCalibration::ConstraintList::value_type & p, input) {
        if (max_depth_difference < 0 || fabs(p.first.depth - p.second.depth) <= max_depth_difference) {
            result.push_back(p);
        }
    }
}

void filterConstraintsByRadiusBinning(const ConstraintList & input, const size_t radius_bin_count, const size_t radius_bin_max_per, ConstraintList & result)
{
    result.clear();
    float max_radius = 1; // assumes scaled

    ConstraintList shuffled_input(input.begin(), input.end());
    std::random_shuffle(shuffled_input.begin(), shuffled_input.end());
    typedef std::map<int, VignetteCalibration::ConstraintList> RadiusBinMap;
    RadiusBinMap radius_bin_map;
    BOOST_FOREACH(const VignetteCalibration::ConstraintList::value_type & p, shuffled_input) {
        float radius_1 = p.first.radius; // just use first radius?
        float radius_2 = p.second.radius;
        float radius = 0.5 * (radius_1 + radius_2);
        int radius_bin = radius / (max_radius / radius_bin_count); // CHECK THIS
        VignetteCalibration::ConstraintList & bin_cl = radius_bin_map[radius_bin];
        if (bin_cl.size() < radius_bin_max_per) {
            bin_cl.push_back(p);
        }
    }

    BOOST_FOREACH(const RadiusBinMap::value_type & m, radius_bin_map) {
        result.insert(result.end(), m.second.begin(), m.second.end());
    }
}

void solveLeastSquaresPolynomial3(const ConstraintList & constraint_list, Eigen::Array3f & result_model)
{
    // M(r) = 1 + a_0 * r^2 + a_1 * r^4 + a_2 * r^6
    Eigen::MatrixXd A(constraint_list.size(), 3);
    Eigen::VectorXd B(constraint_list.size());

    for (size_t i = 0; i < constraint_list.size(); ++i) {
        const Measurement & m1 = constraint_list[i].first;
        const Measurement & m2 = constraint_list[i].second;
        float r1_2 = m1.radius * m1.radius;
        float r1_4 = r1_2 * r1_2;
        float r1_6 = r1_2 * r1_2 * r1_2;
        float r2_2 = m2.radius * m2.radius;
        float r2_4 = r2_2 * r2_2;
        float r2_6 = r2_2 * r2_2 * r2_2;
        float gray_1 = m1.gray_value;
        float gray_2 = m2.gray_value;
        A.row(i)[0] = (r1_2 * gray_2 - r2_2 * gray_1);
        A.row(i)[1] = (r1_4 * gray_2 - r2_4 * gray_1);
        A.row(i)[2] = (r1_6 * gray_2 - r2_6 * gray_1);
        B[i] = gray_1 - gray_2;
    }

    Eigen::VectorXd solution = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);

#if 0
    Eigen::VectorXd householder_solution = A.colPivHouseholderQr().solve(B);
    cout << "householder_solution: " << householder_solution.transpose() << endl;

    Eigen::VectorXd normal_equation_solution = (A.transpose() * A).ldlt().solve(A.transpose()*B);
    cout << "normal equation solution: " << normal_equation_solution.transpose() << endl;
#endif

#if 0
    // look numerically at solution
    {
        cout << "A dimensions: " << A.rows() << "," << A.cols() << endl;
        cout << "B dimensions: " << B.rows() << "," <<  B.cols() << endl;

        // before solution:
        float error_before = (A * Eigen::Vector3d(0,0,0) - B).norm();
        cout << "Error before: " << error_before << endl;

        // after solution:
        float error_after = ((A * solution) - B).norm();
        cout << "Error after: " << error_after << endl;
    }
#endif

    // another type for the solution:
    result_model = Eigen::Array3f(solution[0], solution[1], solution[2]);
}

void solveLeastSquaresPolynomial2(const ConstraintList & constraint_list, Eigen::Array2f & result_model)
{
    // M(r) = 1 + a_0 * r^2 + a_1 * r^4
    Eigen::MatrixXd A(constraint_list.size(), 2);
    Eigen::VectorXd B(constraint_list.size());

    for (size_t i = 0; i < constraint_list.size(); ++i) {
        const Measurement & m1 = constraint_list[i].first;
        const Measurement & m2 = constraint_list[i].second;
        float r1_2 = m1.radius * m1.radius;
        float r1_4 = r1_2 * r1_2;
        float r2_2 = m2.radius * m2.radius;
        float r2_4 = r2_2 * r2_2;
        float gray_1 = m1.gray_value;
        float gray_2 = m2.gray_value;
        A.row(i)[0] = (r1_2 * gray_2 - r2_2 * gray_1);
        A.row(i)[1] = (r1_4 * gray_2 - r2_4 * gray_1);
        B[i] = gray_1 - gray_2;
    }

    Eigen::VectorXd solution = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);

    // another type for the solution:
    result_model = Eigen::Array2f(solution[0], solution[1]);
}

} // ns
