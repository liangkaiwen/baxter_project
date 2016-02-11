#include <boost/format.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include <boost/scoped_ptr.hpp>
#include <boost/foreach.hpp>
#include <boost/timer.hpp>

#include <iostream>
using std::cout;
using std::endl;

#include "params_camera.h"

#include "util.h"

#include "cll.h"
#include "OpenCLAllKernels.h"
#include "ImageBuffer.h"

#include "KernelDepthImageToPoints.h"
#include "KernelTransformPoints.h"
#include "KernelVignetteApplyModelPolynomial3Float.h"
#include "KernelVignetteApplyModelPolynomial3Uchar4.h"

// shouldn't really need this...debugging:
#include "volume_modeler_glfw.h"

// could use other providers at some point
#include "frame_provider_openni2.h"

#include "pose_provider_standard.h"

#include "opencv_utilities.h"

#include "vignette_calibration.h"
using namespace VignetteCalibration;

int main(int argc, char* argv[])
{
    // don't use printf/scanf (and expect it to be in sync)
    std::ios_base::sync_with_stdio(false);

#ifdef _WIN32
#if 1
    // windows: buffer stdout better!
    const int console_buffer_size = 4096;
    char buf[console_buffer_size];
    setvbuf(stdout, buf, _IOLBF, console_buffer_size);
#else
    cout << "WARNING: SLOW COUT" << endl;
#endif
#endif

    // cl path stuff
#ifdef _WIN32
    const fs::path cl_path_default = "C:\\devlibs\\object_modeling\\OpenCLStaticLib";
#else
    const fs::path cl_path_default = "/home/peter/checkout/object_modeling/OpenCLStaticLib";
#endif
    fs::path cl_path;

    fs::path input_oni;
    fs::path input_cameras;
    fs::path output_folder = "output";
    bool pause = false;
    int frame_increment = 1;
    int previous_increment = 20;
    float max_abs_error = -1;
    int radius_bin_count = 20; // number of bins
    int radius_bin_max_per_bin = 20;
    float max_depth_difference = -1;

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("cl_path", po::value<fs::path>(&cl_path), "cl_path")
            ("oni", po::value<fs::path>(&input_oni), "input_oni")
            ("cameras", po::value<fs::path>(&input_cameras), "input_cameras")
            ("output_folder", po::value<fs::path>(&output_folder), "output_folder")
            ("pause", po::value<bool>(&pause)->zero_tokens(), "pause")
            ("frame_increment", po::value<int>(&frame_increment), "frame_increment")
            ("previous_increment", po::value<int>(&previous_increment), "previous_increment")
            ("max_abs_error", po::value<float>(&max_abs_error), "max_abs_error")
            ("radius_bin_max_per_bin", po::value<int>(&radius_bin_max_per_bin), "radius_bin_max_per_bin")
            ("max_depth_difference", po::value<float>(&max_depth_difference), "max_depth_difference")


            // more options
            ;
    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc, po::command_line_style::unix_style ^ po::command_line_style::allow_short), vm);
        po::notify(vm);
    }
    catch (std::exception & e) {
        cout << desc << endl;
        cout << e.what() << endl;
        exit(1);
    }
    if (vm.count("help")) {
        cout << "desc" << endl;
        exit(0);
    }
    if (cl_path.empty()) {
        cl_path = cl_path_default;
    }
    if (!fs::exists(output_folder) && !fs::create_directories(output_folder)) {
        cout << "Couldn't use or create output_folder: " << output_folder << endl;
        exit(1);
    }
    if (input_oni.empty()) {
        cout << "Currently need input_oni" << endl;
        exit(1);
    }
    if (input_cameras.empty()) {
        cout << "Currently need input_cameras" << endl;
        exit(1);
    }


    ///////////////
    // start by initializing opencl
    boost::scoped_ptr<CL> cl_ptr;
    OpenCLPlatformType platform_type = OPENCL_PLATFORM_NVIDIA; //: OPENCL_PLATFORM_INTEL;
    OpenCLContextType context_type = OPENCL_CONTEXT_DEFAULT;
#if 0
    if (cl_intel_cpu) {
        platform_type = OPENCL_PLATFORM_INTEL;
        context_type = OPENCL_CONTEXT_CPU;
    }
    else if (cl_amd_cpu) {
        platform_type = OPENCL_PLATFORM_AMD;
        context_type = OPENCL_CONTEXT_CPU;
    }
#endif
    cl_ptr.reset(new CL(platform_type, context_type));
    if (!cl_ptr->isInitialized()) throw std::runtime_error ("Failed to initialize Open CL");

    boost::shared_ptr<OpenCLAllKernels> all_kernels (new OpenCLAllKernels(*cl_ptr, cl_path));

    // Kernels
    KernelDepthImageToPoints _KernelDepthImageToPoints(*all_kernels);
    KernelTransformPoints _KernelTransformPoints(*all_kernels);

    ///////////// only if you need 3d viewing
    boost::shared_ptr<VolumeModelerGLFW> glfw_ptr;//null
    glfw_ptr.reset(new VolumeModelerGLFW(800,600));
    glfw_ptr->runInThread();

    // params...
    ParamsCamera params_camera;
    params_camera.size = Eigen::Array2i(640,480);
    params_camera.focal = Eigen::Array2f(525,525);
    params_camera.setCenterFromSize();

    // cameras
    PosePtrList camera_list;

    // todo
    // get the frames and cameras
    FrameProviderOpenni2Params frame_provider_params;
    frame_provider_params.file = input_oni;
    boost::shared_ptr<FrameProviderBase> frame_provider_ptr(new FrameProviderOpenni2(frame_provider_params));
    boost::shared_ptr<PoseProviderBase> pose_provider_ptr(new PoseProviderStandard(input_cameras));

    ////////////////
    // must update all of these together
    cv::Mat previous_depth;
    cv::Mat previous_color_bgra;
    Eigen::Affine3f previous_camera_pose = Eigen::Affine3f::Identity();

	// store constraints across frames
    ConstraintList constraint_list_overall;

    cv::Mat color_bgr, depth;
    int frame_counter = -1;
    int frames_processed_counter = -1;

    while (frame_provider_ptr->getNextFrame(color_bgr, depth)) {
        ++frame_counter;

        // always get pose (to keep in sync)
        Eigen::Affine3f camera_pose;
        bool got_pose = pose_provider_ptr->getNextPose(camera_pose);
        if (!got_pose) {
            cout << "Failed to get pose for frame: " << frame_counter << endl;
            break;
        }

        if (frame_counter % frame_increment != 0) continue;
        ++frames_processed_counter;

        cout << "frame: " << frame_counter << " processed: " << frames_processed_counter << endl;

		// prefer bgra
        cv::Mat color_bgra;
        cv::cvtColor(color_bgr, color_bgra, CV_BGR2BGRA);

        ///////////////
        // look at current
        {
            cv::Mat depth_8u;
            const static float depth_factor = 255./5.;
            depth.convertTo(depth_8u, CV_8U, depth_factor);
            cv::Mat depth_bgra;
            cv::cvtColor(depth_8u, depth_bgra, CV_GRAY2BGRA);

            cv::Mat current_both = create1x2(color_bgra, depth_bgra);

            cv::imshow("current_both", current_both);
        }

        /////////////////////////
        // look at projected
        if (!previous_color_bgra.empty()) {
            const int rows = color_bgra.rows;
            const int cols = color_bgra.cols;
            const float radius_scale = (Eigen::Array2f(0,0) - params_camera.center.cast<float>()).matrix().norm();

            // projection transforms, depths, pixels
			Eigen::Affine3f current_to_previous = previous_camera_pose.inverse() * camera_pose;
			cv::Mat current_to_previous_depths;
			cv::Mat current_to_previous_pixels;
			VignetteCalibration::project(all_kernels,
				params_camera,
				current_to_previous,
				depth, 
				current_to_previous_depths,
				current_to_previous_pixels);

            Eigen::Affine3f previous_to_current = camera_pose.inverse() * previous_camera_pose;
            cv::Mat previous_to_current_depths;
            cv::Mat previous_to_current_pixels;
            VignetteCalibration::project(all_kernels,
                params_camera,
                previous_to_current,
                previous_depth,
                previous_to_current_depths,
                previous_to_current_pixels);


            // get gray float
            cv::Mat current_gray;
            cv::Mat previous_gray;
            cv::cvtColor(color_bgra, current_gray, CV_BGRA2GRAY);
            current_gray.convertTo(current_gray, CV_32F, 1./255);
            cv::cvtColor(previous_color_bgra, previous_gray, CV_BGRA2GRAY);
            previous_gray.convertTo(previous_gray, CV_32F, 1./255);


            // get new constraints for this pair
            ConstraintList constraint_list_all_for_pair;

            addConstraints(params_camera, current_to_previous_depths, current_to_previous_pixels, current_gray, previous_depth, previous_gray, constraint_list_all_for_pair);
            // both ways?
            // todo


            ConstraintList cl_max_error;
            filterConstraintsByMaxError(constraint_list_all_for_pair, max_abs_error, cl_max_error);

            ConstraintList cl_value_bound;
            const float min_value = 0.1;
            const float max_value = 0.9;
            filterConstraintsByValue(cl_max_error, min_value, max_value, cl_value_bound);

            ConstraintList cl_depth_difference;
            filterConstraintsByDepthDifference(cl_value_bound, max_depth_difference, cl_depth_difference);

            ConstraintList cl_radius_bin;
            filterConstraintsByRadiusBinning(cl_depth_difference, radius_bin_count, radius_bin_max_per_bin, cl_radius_bin);

            // put filtered from this pair into overall constraints
            constraint_list_overall.insert(constraint_list_overall.end(), cl_radius_bin.begin(), cl_radius_bin.end());

            ////////////
            // look at overall histogram of errors (probably outliers in here)
            {
                const float bin_size = 0.05;
                const int bin_count = 20;
                std::vector<float> error_histogram(bin_count, 0);
                BOOST_FOREACH(const VignetteCalibration::ConstraintList::value_type & p, constraint_list_overall) {
                    int bin = fabs(p.first.gray_value - p.second.gray_value) / bin_size;
                    if (bin >= error_histogram.size()) bin = error_histogram.size() - 1;
                    error_histogram[bin]++;
                }
                cv::Mat histogram_image(50,200,CV_8UC4,cv::Scalar::all(0));
                drawHistogramOnImage(histogram_image, error_histogram, cv::Scalar::all(255));
                cv::imshow("histogram_image", histogram_image);
            }

            if (true) {
                Eigen::Array3f solved_model_polynomial_3;
                solveLeastSquaresPolynomial3(constraint_list_overall, solved_model_polynomial_3);
                cout << "solved_model_polynomial_3: " << solved_model_polynomial_3.transpose() << endl;

				{
                    cv::Mat flat_gray(color_bgra.size(), CV_32FC1, cv::Scalar::all(0.5));
                    cv::Mat flat_gray_vignette = applyVignetteModelPolynomial3<float>(flat_gray, params_camera, solved_model_polynomial_3);
                    cv::imshow("flat_gray_vignette", flat_gray_vignette);
                }

                // replace WITH model
                {
                    cv::Mat_<float> replace_previous_gray_v = applyVignetteModelPolynomial3<float>(previous_gray, params_camera, solved_model_polynomial_3);
                    cv::Mat_<float> replace_current_gray_v = applyVignetteModelPolynomial3<float>(current_gray, params_camera, solved_model_polynomial_3);
                    replaceFromPixels<float>(current_to_previous_depths, current_to_previous_pixels, replace_current_gray_v, replace_previous_gray_v);
                    cv::imshow("replace_previous_gray_v", replace_previous_gray_v);
                }

                // replace WITHOUT model
                {
                    cv::Mat_<float> replace_previous_gray = previous_gray.clone();
                    cv::Mat_<float> replace_current_gray = current_gray.clone();
                    replaceFromPixels<float>(current_to_previous_depths, current_to_previous_pixels, replace_current_gray, replace_previous_gray);
                    cv::imshow("replace_previous_gray", replace_previous_gray);
                }

				// try kernel version (gray)
				if (false) {
					KernelVignetteApplyModelPolynomial3Float _KernelVignetteApplyModelPolynomial3Float(*all_kernels);
					ImageBuffer ib_current_gray(all_kernels->getCL());
					ib_current_gray.setMat(current_gray);
					ImageBuffer ib_current_gray_v(all_kernels->getCL());
					_KernelVignetteApplyModelPolynomial3Float.runKernel(ib_current_gray, ib_current_gray_v, params_camera.center, solved_model_polynomial_3);
					cv::Mat ib_current_gray_v_mat = ib_current_gray_v.getMat();
					cv::imshow("ib_current_gray_v_mat", ib_current_gray_v_mat);

					// compare to cpu version
					cv::Mat cpu_current_gray_v_mat = applyVignetteModelPolynomial3<float>(current_gray, params_camera, solved_model_polynomial_3);
					cv::imshow("cpu_current_gray_v_mat", cpu_current_gray_v_mat);
					cv::Mat diff = cv::abs(cpu_current_gray_v_mat - ib_current_gray_v_mat);
					cv::imshow("diff", diff);
				}

				// try kernel version (color)
				if (false) {

					boost::timer t_1;
					KernelVignetteApplyModelPolynomial3Uchar4 _KernelVignetteApplyModelPolynomial3Uchar4(*all_kernels);
					ImageBuffer ib_current_bgra(all_kernels->getCL());
					ib_current_bgra.setMat(color_bgra);
					ImageBuffer ib_current_bgra_v(all_kernels->getCL());
					_KernelVignetteApplyModelPolynomial3Uchar4.runKernel(ib_current_bgra, ib_current_bgra_v, params_camera.center, solved_model_polynomial_3);
					cv::Mat ib_current_bgra_v_mat = ib_current_bgra_v.getMat();
					cout << "TIME ib_current_bgra_v_mat: " << t_1.elapsed() << endl;
					cv::imshow("ib_current_bgra_v_mat", ib_current_bgra_v_mat);

					// compare to cpu version
					boost::timer t_2;
					cv::Mat cpu_current_bgra_v_mat = applyVignetteModelPolynomial3<cv::Vec4b>(color_bgra, params_camera, solved_model_polynomial_3);
					cout << "TIME cpu_current_bgra_v_mat: " << t_2.elapsed() << endl;
					cv::imshow("cpu_current_bgra_v_mat", cpu_current_bgra_v_mat);
					cv::Mat diff = cv::abs(cpu_current_bgra_v_mat - ib_current_bgra_v_mat);
					cv::imshow("diff", diff);
				}
            }


            if (false) {
                Eigen::Array2f solved_model_polynomial_2;
                solveLeastSquaresPolynomial2(constraint_list_overall, solved_model_polynomial_2);
                cout << "solved_model_polynomial_2: " << solved_model_polynomial_2.transpose() << endl;


				{
                    cv::Mat flat_gray(color_bgra.size(), CV_32FC1, cv::Scalar::all(0.5));
                    cv::Mat flat_gray_vignette = VignetteCalibration::applyVignetteModelPolynomial2<float>(flat_gray, params_camera, solved_model_polynomial_2);
                    cv::imshow("flat_gray_vignette", flat_gray_vignette);
                }


            }



        }


		// can use 3d viewer if you want...
#if 0
        // look at current in world?
        // only time you actually need world points
        {
            ImageBuffer ib_points_in_world(all_kernels->getCL());
            _KernelTransformPoints.runKernel(ib_points, ib_points_in_world, camera_pose);
            MeshVertexVectorPtr vertices(new MeshVertexVector);
            cv::Mat points_mat = ib_points_in_world.getMat();
            getVerticesForPointsAndColors(points_mat, color_bgra, *vertices);
            glfw_ptr->updatePointCloud("current_in_world", vertices);
        }
#endif


        // look at cv images
        int key = -1;
        if (pause) {
            cout << "pause..." << endl;
            key = cv::waitKey(0);
        }
        else {
            key = cv::waitKey(10);
        }
        if (glfw_ptr) {
            int glfw_key = glfw_ptr->getKeyLowerCaseSync();
            if (glfw_key > 0) {
                key = glfw_key;
                glfw_ptr->clearKeySync();
            }
        }

        // act on key
        if (key == 'q') break;

        ////////////////////
        // update previous for next time
#if 0
        // must update all of these together
        ImageBuffer previous_ib_color(all_kernels->getCL());
        ImageBuffer previous_ib_depth(all_kernels->getCL());
        ImageBuffer previous_ib_points(all_kernels->getCL());
        cv::Mat previous_color_bgra;
        Eigen::Affine3f previous_camera_pose = Eigen::Affine3f::Identity();
#endif
        if (frames_processed_counter % previous_increment == 0) {
            previous_depth = depth;
            previous_color_bgra = color_bgra;
            previous_camera_pose = camera_pose;
        }
    }

    cout << "pause at end..." << endl;
    cv::waitKey(0);

    // exit cleanly
    if (glfw_ptr) {
        glfw_ptr->destroy();
        glfw_ptr->join();
    }

    return 0;
}
