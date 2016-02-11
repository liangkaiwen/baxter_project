#include "model_histogram.h"

#include "EigenUtilities.h"

#include "opencv_utilities.h"

#include "util.h"

ModelHistogram::ModelHistogram(boost::shared_ptr<OpenCLAllKernels> all_kernels, VolumeModelerAllParams const& params, boost::shared_ptr<Alignment> alignment_ptr, RenderBuffers const& render_buffers)
    : ModelBase(all_kernels, params, alignment_ptr, render_buffers),
      kernel_add_frame_to_histogram_(*all_kernels),
      kernel_histogram_sum_(*all_kernels),
      kernel_histogram_sum_check_index_(*all_kernels),
      kernel_divide_floats_(*all_kernels),
      kernel_extract_volume_slice_(*all_kernels),
      kernel_extract_volume_float_(*all_kernels),
      kernel_render_points_and_normals_(*all_kernels),
      kernel_histogram_max_(*all_kernels),
      kernel_histogram_max_check_index_(*all_kernels),
      kernel_histogram_variance_(*all_kernels),
      kernel_add_floats_(*all_kernels),
      kernel_gaussian_pdf_(*all_kernels),
      kernel_gaussian_pdf_constant_x_(*all_kernels)
{
    reset();
}

ModelHistogram::ModelHistogram(ModelHistogram const& other)
    : ModelBase(other),
      kernel_add_frame_to_histogram_(other.kernel_add_frame_to_histogram_),
      kernel_histogram_sum_(other.kernel_histogram_sum_),
      kernel_histogram_sum_check_index_(other.kernel_histogram_sum_check_index_),
      kernel_divide_floats_(other.kernel_divide_floats_),
      kernel_extract_volume_slice_(other.kernel_extract_volume_slice_),
      kernel_extract_volume_float_(other.kernel_extract_volume_float_),
      kernel_render_points_and_normals_(other.kernel_render_points_and_normals_),
      kernel_histogram_max_(other.kernel_histogram_max_),
      kernel_histogram_max_check_index_(other.kernel_histogram_max_check_index_),
      kernel_histogram_variance_(other.kernel_histogram_variance_),
      kernel_add_floats_(other.kernel_add_floats_),
      kernel_gaussian_pdf_(other.kernel_gaussian_pdf_),
      kernel_gaussian_pdf_constant_x_(other.kernel_gaussian_pdf_constant_x_)
{
    // todo?
}

ModelHistogram* ModelHistogram::clone()
{
    return new ModelHistogram(*this);
}

void ModelHistogram::reset()
{
	volume_buffer_list_.resize(params_.model_histogram.bin_count);
	for(size_t i = 0; i < volume_buffer_list_.size(); ++i) {
		volume_buffer_list_[i].reset(new VolumeBuffer(all_kernels_));
		volume_buffer_list_[i]->resize(params_.volume.cell_count, sizeof(float));
		volume_buffer_list_[i]->setFloat(0);
	}

	volume_pose_ = Eigen::Affine3f::Identity();
}


void ModelHistogram::getMaxAndIndex(VolumeBuffer & max_value, VolumeBuffer & max_index)
{
    max_value.resize(volume_buffer_list_[0]->getVolumeCellCounts(), sizeof(float));
    max_value.setFloat(0);
    max_index.resize(volume_buffer_list_[0]->getVolumeCellCounts(), sizeof(int));
    max_index.setInt(-1);

    for (size_t bin = 0; bin < params_.model_histogram.bin_count; ++bin) {
        float bin_min, bin_max;
        getMinMaxBinValues(bin, bin_min, bin_max);

        kernel_histogram_max_.runKernel(*volume_buffer_list_[bin], max_index, max_value, (int) bin);
    }
}

void ModelHistogram::getMaxAndIndexOutsideIndex(VolumeBuffer & input_indices, int range, VolumeBuffer & max_value, VolumeBuffer & max_index)
{
    max_value.resize(volume_buffer_list_[0]->getVolumeCellCounts(), sizeof(float));
    max_value.setFloat(0);
    max_index.resize(volume_buffer_list_[0]->getVolumeCellCounts(), sizeof(int));
    max_index.setInt(-1);

    for (size_t bin = 0; bin < params_.model_histogram.bin_count; ++bin) {
        float bin_min, bin_max;
        getMinMaxBinValues(bin, bin_min, bin_max);

        kernel_histogram_max_check_index_.runKernel(*volume_buffer_list_[bin], input_indices, max_index, max_value, (int) bin, range);
    }
}

void ModelHistogram::computeMean(VolumeBuffer & mean, VolumeBuffer & count)
{
    mean.resize(volume_buffer_list_[0]->getVolumeCellCounts(), sizeof(float));
    mean.setFloat(0);
    count.resize(volume_buffer_list_[0]->getVolumeCellCounts(), sizeof(float));
    count.setFloat(0);

    for (size_t bin = 0; bin < params_.model_histogram.bin_count; ++bin) {
        float bin_min, bin_max;
        getMinMaxBinValues(bin, bin_min, bin_max);

        kernel_histogram_sum_.runKernel(*volume_buffer_list_[bin], mean, count, bin_min, bin_max);
    }

    // stupid linux
    cl::Buffer mean_buffer = mean.getBuffer();
    cl::Buffer count_buffer = count.getBuffer();
    kernel_divide_floats_.runKernel(mean_buffer, count_buffer, mean.getSizeInCells());
}

void ModelHistogram::computeMeanAroundIndex(VolumeBuffer & input_indices, int range, VolumeBuffer & mean, VolumeBuffer & count)
{
    mean.resize(volume_buffer_list_[0]->getVolumeCellCounts(), sizeof(float));
    mean.setFloat(0);
    count.resize(volume_buffer_list_[0]->getVolumeCellCounts(), sizeof(float));
    count.setFloat(0);

    for (size_t bin = 0; bin < params_.model_histogram.bin_count; ++bin) {
        float bin_min, bin_max;
        getMinMaxBinValues(bin, bin_min, bin_max);

        kernel_histogram_sum_check_index_.runKernel(*volume_buffer_list_[bin], input_indices, mean, count, (int)bin, range, bin_min, bin_max);
    }

    // stupid linux
    cl::Buffer mean_buffer = mean.getBuffer();
    cl::Buffer count_buffer = count.getBuffer();
    kernel_divide_floats_.runKernel(mean_buffer, count_buffer, mean.getSizeInCells());
}


void ModelHistogram::extractPeakAndMeanAround(VolumeBuffer & max_index, VolumeBuffer & max_value, VolumeBuffer & mean, VolumeBuffer & count)
{
    getMaxAndIndex(max_value, max_index);
    computeMeanAroundIndex(max_index, params_.model_histogram.peak_finding_bin_range, mean, count);
}


void ModelHistogram::extractPeakAndMeanAroundSecondPeak(VolumeBuffer & input_indices, VolumeBuffer & max_index, VolumeBuffer & max_value, VolumeBuffer & mean, VolumeBuffer & count)
{
    getMaxAndIndexOutsideIndex(input_indices, params_.model_histogram.peak_finding_bin_range, max_value, max_index);
    computeMeanAroundIndex(max_index, params_.model_histogram.peak_finding_bin_range, mean, count);;
}

void ModelHistogram::computeMeanAndVariance(VolumeBuffer & result_mean, VolumeBuffer & result_count, VolumeBuffer & result_variance)
{
    computeMean(result_mean, result_count);

    result_variance.resize(volume_buffer_list_[0]->getVolumeCellCounts(), sizeof(float));
    result_variance.setFloat(0);
    for (size_t bin = 0; bin < params_.model_histogram.bin_count; ++bin) {
        float bin_min, bin_max;
        getMinMaxBinValues(bin, bin_min, bin_max);

        // now variance kernel...
        kernel_histogram_variance_.runKernel(*volume_buffer_list_[bin], result_mean, result_count, result_variance, bin_min, bin_max);
    }
}

void ModelHistogram::computeProbabilityOfZeroBins(VolumeBuffer & probability_of_zero_bins)
{
    // add up all counts
    VolumeBuffer all_counts(all_kernels_);
    all_counts.resize(volume_buffer_list_[0]->getVolumeCellCounts(), sizeof(float));
    all_counts.setFloat(0);

    // add all bins for total counts
    for (size_t bin = 0; bin < params_.model_histogram.bin_count; ++bin) {
        cl::Buffer all_counts_buffer = all_counts.getBuffer();
        cl::Buffer this_bin_buffer = volume_buffer_list_[bin]->getBuffer();
        kernel_add_floats_.runKernel(all_counts_buffer, this_bin_buffer, all_counts.getSizeInCells());
    }

    // now add the two central bins? (or one bin if odd bin number?)
    //VolumeBuffer zero_bin_counts(*all_kernels_);
    probability_of_zero_bins.resize(volume_buffer_list_[0]->getVolumeCellCounts(), sizeof(float));
    probability_of_zero_bins.setFloat(0);
    for (size_t bin = (params_.model_histogram.bin_count - 1) / 2; bin <= params_.model_histogram.bin_count / 2; ++bin) {
        cl::Buffer probability_of_zero_bins_buffer = probability_of_zero_bins.getBuffer();
        cl::Buffer this_bin_buffer = volume_buffer_list_[bin]->getBuffer();
        kernel_add_floats_.runKernel(probability_of_zero_bins_buffer, this_bin_buffer, probability_of_zero_bins.getSizeInCells());
    }

    // now divide
    cl::Buffer probability_of_zero_bins_buffer = probability_of_zero_bins.getBuffer();
    cl::Buffer all_counts_buffer = all_counts.getBuffer();
    kernel_divide_floats_.runKernel(probability_of_zero_bins_buffer, all_counts_buffer, probability_of_zero_bins.getSizeInCells());
}


void ModelHistogram::computerGaussianPDFForZero(VolumeBuffer & means, VolumeBuffer & variances, VolumeBuffer & pdf_values)
{
    pdf_values.resize(means.getVolumeCellCounts(), sizeof(float));

    // stupid linux
    cl::Buffer means_buffer = means.getBuffer();
    cl::Buffer variances_buffer = variances.getBuffer();
    cl::Buffer pdf_values_buffer = pdf_values.getBuffer();

    kernel_gaussian_pdf_constant_x_.runKernel(means_buffer, variances_buffer, 0, pdf_values_buffer, means.getSizeInCells());
}


void ModelHistogram::renderModel(
        const ParamsCamera & params_camera,
        const Eigen::Affine3f & model_pose,
        RenderBuffers & render_buffers)
{
    render_buffers.setSize(params_camera.size.x(), params_camera.size.y());
    render_buffers.resetAllBuffers();


    // experiment with rendering methods...

#if 0
    // first is just get the mean and render that
    {
        VolumeBuffer sum(*all_kernels_);
        sum.resize(volume_buffer_list_[0]->getVolumeCellCounts(), sizeof(float));
        sum.setFloat(0, false);
        VolumeBuffer count(*all_kernels_);
        count.resize(volume_buffer_list_[0]->getVolumeCellCounts(), sizeof(float));
        count.setFloat(0, false);

        for (size_t bin = 0; bin < params_.model_histogram.bin_count; ++bin) {
            float bin_min, bin_max;
            getMinMaxBinValues(bin, bin_min, bin_max);

            kernel_histogram_sum_.runKernel(*volume_buffer_list_[bin], sum, count, bin_min, bin_max, false);
        }

        // stupid linux
        cl::Buffer sum_buffer = sum.getBuffer();
        cl::Buffer count_buffer = count.getBuffer();

        kernel_divide_floats_.runKernel(sum_buffer, count_buffer, sum.getCellCountTotal(), true);


        // now have mean values in "sum"
        // can use count as weight...

        VolumeBuffer volume_d = sum;
        VolumeBuffer volume_dw = count;

        // stupid linux
        ImageBuffer image_buffer_mask = render_buffers.getImageBufferMask();
        ImageBuffer image_buffer_points = render_buffers.getImageBufferPoints();
        ImageBuffer image_buffer_normals = render_buffers.getImageBufferNormals();


        kernel_render_points_and_normals_.runKernel(volume_d,
                                                    volume_dw,
                                                    image_buffer_mask,
                                                    image_buffer_points,
                                                    image_buffer_normals,
                                                    params_.volume.cell_size,
                                                    pose * volume_pose_,
                                                    params_camera.focal,
                                                    params_camera.center,
                                                    params_camera.min_max_depth[0],
                                                    params_camera.min_max_depth[1],
                                                    mask_value);
    }
#endif

    /////////////////////////
    // next try extracting the peak value(s) ala Dieter...
    {
        VolumeBuffer max_index(all_kernels_);
        VolumeBuffer max_value(all_kernels_);
        VolumeBuffer mean(all_kernels_);
        VolumeBuffer count(all_kernels_);
        extractPeakAndMeanAround(max_index, max_value, mean, count);

        // try rendering these values?
        VolumeBuffer volume_d = mean;
        VolumeBuffer volume_dw = count;

        {
            const int mask_value = 1;
            kernel_render_points_and_normals_.runKernel(volume_d,
                                                        volume_dw,
                                                        render_buffers.getImageBufferMask(),
                                                        render_buffers.getImageBufferPoints(),
                                                        render_buffers.getImageBufferNormals(),
                                                        params_.volume.cell_size,
                                                        model_pose * volume_pose_,
                                                        params_camera.focal,
                                                        params_camera.center,
                                                        params_camera.min_max_depth[0],
                                                        params_camera.min_max_depth[1],
                                                        mask_value);

        }
    }


    // earlier? elsewhere?
    //PickPixel pick_pixel_render("PickPixelRender");
    // get depth map for picking?

}

void ModelHistogram::updateModel(
        Frame & frame,
        const Eigen::Affine3f & model_pose)
{
	// do this before base class because won't be empty after that
	if (isEmpty()) {
		volume_pose_ = getSingleVolumePoseForFirstFrame(frame);
	}

	ModelBase::updateModel(frame, model_pose);

	ImageBuffer buffer_segments(all_kernels_->getCL());

	Eigen::Affine3f volume_pose_in_world = model_pose * volume_pose_;

	for (size_t bin = 0; bin < params_.model_histogram.bin_count; ++bin) {
		float bin_min, bin_max;
		getMinMaxBinValues(bin, bin_min, bin_max);
		
		kernel_add_frame_to_histogram_.runKernel(*volume_buffer_list_[bin], frame.image_buffer_depth, frame.image_buffer_segments, 0,
            bin_min, bin_max, params_.volume.cell_size, volume_pose_in_world, params_.camera.focal, params_.camera.center, params_.volume.min_truncation_distance);
	}

	///// now debug
	const int debug = true;
    if (debug) {
		// now also extract a middle slice for each bin and dump as image...
#if 0
		for (size_t bin = 0; bin < params_.model_histogram.bin_count; ++bin) {
			float bin_min, bin_max;
			getMinMaxBinValues(bin, bin_min, bin_max);

			int axis = 1;
			int position = volume_buffer_list_[0]->getVolumeCellCounts()[axis] / 2;

			ImageBuffer slice(all_kernels_->getCL());
			kernel_extract_volume_slice_.runKernel(*volume_buffer_list_[bin], axis, position, slice);

			cv::Mat slice_mat = slice.getMat();
			// todo: scale smartly
			cv::imshow("slice_mat", slice_mat);
			
			cout << "slice_mat pause..." << endl;
			cv::waitKey(0);
		}
#endif

		// pick pixel here?
		// could put this part in model base for example?
		// or even within pick pixel itself if it were aware of depth
		// duplicated for kmeans
		if (pick_pixel_) {
			cv::Point2i pixel_cv = pick_pixel_->getPixel();
			Eigen::Array2i pixel_eigen(pixel_cv.x, pixel_cv.y);
			if ((pixel_eigen != last_pick_pixel_).any() && (pixel_eigen >= 0).all()) {
				last_pick_pixel_ = pixel_eigen;
                float depth_at_pick_pixel = frame.mat_depth.at<float>(last_pick_pixel_[1], last_pick_pixel_[0]);
                depth_at_pick_pixel -= params_.model_histogram.debug_pick_pixel_depth_offset;
                last_pick_pixel_world_point_ = model_pose.inverse() * EigenUtilities::depthToPoint(params_.camera.focal, params_.camera.center, last_pick_pixel_, depth_at_pick_pixel);
                last_pick_pixel_camera_point_ = model_pose.inverse() * Eigen::Vector3f(0,0,0);
            }
        }

        // get histograms for points along ray (including last_pick_pixel_world_point_)
        {

            // all points along debug ray
            {
                Eigen::Vector3f ray_to_point = (last_pick_pixel_world_point_ - last_pick_pixel_camera_point_);
                float t_for_point = ray_to_point.norm();
                ray_to_point /= t_for_point;

                float t_step = params_.volume.cell_size;

                MeshVertexVectorPtr mesh_vertices_on_ray(new MeshVertexVector);
                std::vector<cv::Mat> histogram_image_list;

#if 0
                // also peaks here?
                VolumeBuffer max_index(*all_kernels_);
                VolumeBuffer max_value(*all_kernels_);
                VolumeBuffer peak_mean(*all_kernels_);
                VolumeBuffer peak_count(*all_kernels_);
                extractPeakAndMeanAround(max_index, max_value, peak_mean, peak_count);
                std::vector<cv::Mat> peak_image_list;
#endif

                // also gaussian(s)
                VolumeBuffer means(all_kernels_);
                VolumeBuffer counts(all_kernels_);
                VolumeBuffer variances(all_kernels_);
                computeMeanAndVariance(means, counts, variances);
                std::vector<cv::Mat> gaussians_image_list;

                // also probability of zero bin (compare to gaussian PDF?)
                VolumeBuffer prob_of_zero_bins(all_kernels_);
                computeProbabilityOfZeroBins(prob_of_zero_bins);


                for (int step = -params_.model_histogram.debug_points_along_ray; step <= params_.model_histogram.debug_points_along_ray; ++step) {
                    Eigen::Vector3f point_on_ray = last_pick_pixel_camera_point_+ (t_for_point + t_step * step) * ray_to_point;
                    Eigen::Vector3f point_in_volume = volume_pose_.inverse() * point_on_ray;
                    Eigen::Array3i nearest_voxel = EigenUtilities::roundPositiveVector3fToInt((point_in_volume / params_.volume.cell_size)).array();


                    // add points to point cloud
                    MeshVertex vertex;
                    vertex.p.head<3>() = point_on_ray;
                    vertex.p[3] = 1;
                    vertex.c = Eigen::Array4ub(0,0,255,255);
                    mesh_vertices_on_ray->push_back(vertex);

                    int width = params_.model_histogram.mat_width_per_bin * params_.model_histogram.bin_count;
                    int height = params_.model_histogram.mat_height;
                    float min_value = params_.model_histogram.min_value;
                    float max_value = params_.model_histogram.max_value;

                    // grab histogram
                    std::vector<float> histogram;
                    getHistogramForPoint(point_on_ray, histogram);

                    ////////////////////////
                    // add histogram image to list
                    cv::Mat histogram_image(height, width, CV_8UC3, cv::Scalar::all(0));
                    //histogram_image_list.push_back(drawHistogramImage(histogram, params_.model_histogram.mat_width_per_bin, params_.model_histogram.mat_height, true, true));
                    drawHistogramOnImage(histogram_image, histogram, cv::Scalar::all(255));
                    drawVerticalCenterLine(histogram_image);
                    // also probability of zero bins
                    if (isVertexInVolume(prob_of_zero_bins.getVolumeCellCounts(), nearest_voxel)) {
                        float prob_zero_bins;
                        kernel_extract_volume_float_.runKernel(prob_of_zero_bins, nearest_voxel, prob_zero_bins);
                        drawVerticalLine(histogram_image, min_value, max_value, 0, prob_zero_bins, 1, cv::Scalar(0,255,0));
                    }
                    histogram_image_list.push_back(getImageWithBorder(histogram_image));


#if 0
                    ///////////////////////
                    // peaks
                    cv::Mat peaks_image(height, width, CV_8UC3, cv::Scalar::all(0));
                    if (isVertexInVolume(peak_mean.getVolumeCellCounts(), nearest_voxel)) {
                        float peak_mean_value;
                        kernel_extract_volume_float_.runKernel(peak_mean, nearest_voxel, peak_mean_value);
                        //float peak_count;
                        //kernel_extract_volume_float_.runKernel(peak_count, nearest_voxel, peak_count);
                        drawVerticalLine(peaks_image, min_value, max_value, peak_mean_value, 1.0, 0, cv::Scalar::all(255));
                    }
                    drawVerticalCenterLine(peaks_image);
                    peak_image_list.push_back(getImageWithBorder(peaks_image));
#endif

                    //////////////////////
                    // gaussians
                    cv::Mat gaussian_image(height, width, CV_8UC3, cv::Scalar::all(0));
                    drawVerticalCenterLine(gaussian_image);
                    if (isVertexInVolume(means.getVolumeCellCounts(), nearest_voxel)){
                        float gaussian_mean, gaussian_variance;
                        kernel_extract_volume_float_.runKernel(means, nearest_voxel, gaussian_mean);
                        kernel_extract_volume_float_.runKernel(variances, nearest_voxel, gaussian_variance);
                        drawGaussianOnImage(gaussian_image, min_value, max_value, gaussian_mean, gaussian_variance, params_.model_histogram.bin_count);

                        // now a thick line for the probability of zero in this gaussian
                        float pdf_at_0 = gaussianPDF(gaussian_mean, gaussian_variance, 0);
                        // hack in
                        pdf_at_0 /= 100;
                        drawVerticalLine(gaussian_image, min_value, max_value, 0, pdf_at_0, 1, cv::Scalar(0,255,0));
                    }
                    gaussians_image_list.push_back(getImageWithBorder(gaussian_image));

                }

                if (update_interface_) update_interface_->updatePointCloud(debug_string_prefix_ + "mesh_vertices_on_ray", mesh_vertices_on_ray);

                // build and show image
                // this could still crash...
                if (!histogram_image_list.empty() && !histogram_image_list[0].empty()) {
                    cv::Mat all_histogram_mats = createMxN(histogram_image_list.size(), 1, histogram_image_list);
                    cv::imshow(debug_string_prefix_ + "all_histogram_mats", all_histogram_mats);
                }

#if 0
                // build and show image
                // this could probably crash somehow...
                if (!peak_image_list.empty() && !peak_image_list[0].empty()) {
                    cv::Mat all_peaks = createMxN(peak_image_list.size(), 1, peak_image_list);
                    cv::imshow(debug_string_prefix_ + "peak_image_list", all_peaks);
                }
#endif

                // again build and show
                if (!gaussians_image_list.empty() && !gaussians_image_list[0].empty()) {
                    cv::Mat all_gaussians = createMxN(gaussians_image_list.size(), 1, gaussians_image_list);
                    cv::imshow(debug_string_prefix_ + "gaussians_image_list", all_gaussians);
                }


            }

        }
    }
}

void ModelHistogram::generateMesh(MeshVertexVector & vertex_list, TriangleVector & triangle_list)
{
    // todo?
}

void ModelHistogram::generateMeshAndValidity(MeshVertexVector & vertex_list, TriangleVector & triangle_list, std::vector<bool> & vertex_validity, std::vector<bool> & triangle_validity)
{
    // todo?
}

void ModelHistogram::deallocateBuffers()
{
    // todo?
}

void ModelHistogram::save(fs::path const& folder)
{
    ModelBase::save(folder);

    // more?
}

void ModelHistogram::load(fs::path const& folder)
{
    ModelBase::load(folder);

    // more?
}


void ModelHistogram::getBoundingLines(MeshVertexVector & vertex_list)
{
	Eigen::Vector4ub color(0,0,255,0);
	::getBoundingLines(volume_buffer_list_[0]->getVolumeCellCounts(), params_.volume.cell_size, volume_pose_, color, vertex_list);
}


void ModelHistogram::refreshUpdateInterface()
{
	if (update_interface_) {
		MeshVertexVectorPtr bounding_lines_ptr (new MeshVertexVector);
		getBoundingLines(*bounding_lines_ptr);
		update_interface_->updateLines("ModelHistogram", bounding_lines_ptr);
	}
}

void ModelHistogram::getMinMaxBinValues(size_t which_bin, float & result_min, float & result_max)
{
	if (which_bin >= params_.model_histogram.bin_count) { cout << "nope" << endl; exit(1);}

	float range = params_.model_histogram.max_value - params_.model_histogram.min_value;
	float range_per_bin = range / (float) params_.model_histogram.bin_count;
	result_min = params_.model_histogram.min_value + which_bin * range_per_bin;
	result_max = result_min + range_per_bin;
}

bool ModelHistogram::getHistogramForVoxel(const Eigen::Array3i & voxel, std::vector<float> & result_histogram)
{
    result_histogram.assign(params_.model_histogram.bin_count, 0);
	if (!isVertexInVolume(volume_buffer_list_[0]->getVolumeCellCounts(), voxel)) return false;

	for (size_t bin = 0; bin < params_.model_histogram.bin_count; ++bin) {
		float bin_min, bin_max;
		getMinMaxBinValues(bin, bin_min, bin_max);

		float value;
		kernel_extract_volume_float_.runKernel(*volume_buffer_list_[bin], voxel, value);

        result_histogram[bin] = value;
	}

	return true;
}
	
bool ModelHistogram::getHistogramForPoint(const Eigen::Vector3f & world_point, std::vector<float> & result_histogram)
{
	Eigen::Vector3f point_in_volume = volume_pose_.inverse() * world_point;

	// this may put slightly too negative points into the voxel grid...ok for now...do correct rounding at some point
	Eigen::Array3i nearest_voxel = EigenUtilities::roundPositiveVector3fToInt((point_in_volume / params_.volume.cell_size)).array();

	return getHistogramForVoxel(nearest_voxel, result_histogram);
}
