#pragma once

#include "model_base.h"

#include "pick_pixel.h"

#include "VolumeBuffer.h"
#include <vector>

#include "KernelAddFrameToHistogram.h"
#include "KernelHistogramSum.h"
#include "KernelHistogramSumCheckIndex.h"
#include "KernelDivideFloats.h"
#include "KernelExtractVolumeSlice.h"
#include "KernelExtractVolumeFloat.h"
#include "KernelRenderPointsAndNormals.h"
#include "KernelHistogramMax.h"
#include "KernelHistogramMaxCheckIndex.h"
#include "KernelHistogramVariance.h"
#include "KernelAddFloats.h"
#include "KernelGaussianPDF.h"
#include "KernelGaussianPDFConstantX.h"

class ModelHistogram : public ModelBase
{
public:
    ModelHistogram(boost::shared_ptr<OpenCLAllKernels> all_kernels, VolumeModelerAllParams const& params, boost::shared_ptr<Alignment> alignment_ptr, RenderBuffers const& render_buffers);
    ModelHistogram(ModelHistogram const& other);

    virtual ModelHistogram* clone();

	virtual void reset();

	// inherited

    virtual void renderModel(
            const ParamsCamera & params_camera,
            const Eigen::Affine3f & model_pose,
            RenderBuffers & render_buffers);

	virtual void updateModel(
		Frame & frame,
		const Eigen::Affine3f & model_pose);

	virtual void generateMesh(
		MeshVertexVector & vertex_list,
		TriangleVector & triangle_list);

	virtual void generateMeshAndValidity(
		MeshVertexVector & vertex_list, 
		TriangleVector & triangle_list, 
		std::vector<bool> & vertex_validity, 
		std::vector<bool> & triangle_validity);

    virtual void deallocateBuffers();

	virtual void save(fs::path const& folder);

	virtual void load(fs::path const& folder);

	virtual void refreshUpdateInterface();

    virtual void getBoundingLines(MeshVertexVector & vertex_list);

    ///////////////
    // unique to this model:

	virtual bool getHistogramForVoxel(const Eigen::Array3i & voxel, std::vector<float> & result_histogram);
	
	virtual bool getHistogramForPoint(const Eigen::Vector3f & world_point, std::vector<float> & result_histogram);

    virtual void getMaxAndIndex(VolumeBuffer & max_value, VolumeBuffer & max_index);

    virtual void getMaxAndIndexOutsideIndex(VolumeBuffer & input_indices, int range, VolumeBuffer & max_value, VolumeBuffer & max_index);

    virtual void computeMean(VolumeBuffer & mean, VolumeBuffer & count);

    virtual void computeMeanAroundIndex(VolumeBuffer & input_indices, int range, VolumeBuffer & mean, VolumeBuffer & count);

    // The Dieter peak idea:
    virtual void extractPeakAndMeanAround(VolumeBuffer & max_index, VolumeBuffer & max_value, VolumeBuffer & mean, VolumeBuffer & count);
    virtual void extractPeakAndMeanAroundSecondPeak(VolumeBuffer & input_indices, VolumeBuffer & max_index, VolumeBuffer & max_value, VolumeBuffer & mean, VolumeBuffer & count);

    // find variance?  Seems generally useful
    virtual void computeMeanAndVariance(VolumeBuffer & result_mean, VolumeBuffer & result_count, VolumeBuffer & result_variance);

    virtual void computeProbabilityOfZeroBins(VolumeBuffer & probability_of_zero_bins);

    virtual void computerGaussianPDFForZero(VolumeBuffer & means, VolumeBuffer & variances, VolumeBuffer & pdf_values);


protected:
	// functions

	void getMinMaxBinValues(size_t which_bin, float & result_min, float & result_max);

	// members:
    
	std::vector<boost::shared_ptr<VolumeBuffer> > volume_buffer_list_;
	Eigen::Affine3f volume_pose_;

	// duplicated in models
	Eigen::Array2i last_pick_pixel_;
	Eigen::Vector3f last_pick_pixel_world_point_;
    Eigen::Vector3f last_pick_pixel_camera_point_;

	// kernels needed here?
	KernelAddFrameToHistogram kernel_add_frame_to_histogram_;
	KernelHistogramSum kernel_histogram_sum_;
    KernelHistogramSumCheckIndex kernel_histogram_sum_check_index_;
	KernelDivideFloats kernel_divide_floats_;
	KernelExtractVolumeSlice kernel_extract_volume_slice_;
	KernelExtractVolumeFloat kernel_extract_volume_float_;
    KernelRenderPointsAndNormals kernel_render_points_and_normals_;
    KernelHistogramMax kernel_histogram_max_;
    KernelHistogramMaxCheckIndex kernel_histogram_max_check_index_;
    KernelHistogramVariance kernel_histogram_variance_;
    KernelAddFloats kernel_add_floats_;
    KernelGaussianPDF kernel_gaussian_pdf_;
    KernelGaussianPDFConstantX kernel_gaussian_pdf_constant_x_;


public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
