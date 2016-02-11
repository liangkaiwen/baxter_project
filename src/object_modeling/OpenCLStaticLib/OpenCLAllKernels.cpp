#include "stdafx.h"
#include "OpenCLAllKernels.h"

#include "OpenCLKernelsBuilder.h"

// Can include these to make sure string name is only one place
#include "KernelSetFloat.h"
#include "KernelSetUChar.h"
#include "KernelSetInt.h"
#include "KernelSetFloat4.h"
#include "KernelAddFrameToHistogram.h"
#include "KernelHistogramSum.h"
#include "KernelHistogramSumCheckIndex.h"
#include "KernelHistogramMax.h"
#include "KernelHistogramMaxCheckIndex.h"
#include "KernelHistogramVariance.h"
#include "KernelDivideFloats.h"

#include "KernelExtractVolumeSlice.h"
#include "KernelExtractVolumeSliceFloat4.h"
#include "KernelExtractVolumeSliceFloat4Length.h"
#include "KernelExtractVolumeFloat.h"

#include "KernelAddFrame.h"
#include "KernelAddFrameTo2Means.h"
#include "KernelAddFrameTo2MeansUsingNormals.h"
#include "KernelAddFrameTo2MeansUsingStoredNormals.h"
#include "KernelAddFrameIfCompatible.h"

#include "KernelRenderPointsAndNormals.h"
#include "KernelRenderPoints.h"
#include "KernelRenderNormalForPoints.h"
#include "KernelRenderColorForPoints.h"
#include "KernelRenderMax.h"
#include "KernelRender2MeansAbs.h"

#include "KernelNormalsToShadedImage.h"
#include "KernelNormalsToColorImage.h"
#include "KernelPointsToDepthImage.h"
#include "KernelAddFloats.h"
#include "KernelAddFloatsWithWeights.h"
#include "KernelAddFloatsWithWeightsExternalWeight.h"

#include "KernelGaussianPDF.h"
#include "KernelGaussianPDFConstantX.h"
#include "KernelDotVolumeNormal.h"
#include "KernelMinAbsVolume.h"
#include "KernelAddVolumes.h"
#include "KernelBetterNormal.h"
#include "KernelExtractFloat4ForPointImage.h"
#include "KernelExtractIntForPointImage.h"
#include "KernelApplyPoseToNormals.h"
#include "KernelApplyPoseToPoints.h"

#include "KernelMinAbsFloatsWithWeights.h"
#include "KernelMinAbsFloatsWithWeightsRecordIndex.h"
#include "KernelMinAbsFloatsWithWeightsAndMinimumWeightFraction.h"

#include "KernelComputeNormalVolume.h"
#include "KernelComputeNormalVolumeWithWeights.h"
#include "KernelComputeNormalVolumeWithWeightsUnnormalized.h"

#include "KernelMaxFloats.h"
#include "KernelMinFloats.h"
#include "KernelPickIfIndexFloats.h"
#include "KernelPickIfIndexFloat4.h"

#include "KernelSetVolumeSDFBox.h"
#include "KernelRaytraceBox.h"
#include "KernelRaytraceSpecial.h"

#include "KernelMarkPointsViolateEmpty.h"

#include "KernelDepthImageToPoints.h"
#include "KernelTransformPoints.h"
#include "KernelSetInvalidPointsTrue.h"

#include "KernelOptimizeErrorAndJacobianICP.h"
#include "KernelOptimizeErrorAndJacobianImage.h"
#include "KernelOptimizeNormalEquationTerms.h"

#include "KernelVignetteApplyModelPolynomial3Uchar4.h"
#include "KernelVignetteApplyModelPolynomial3Float.h"

#include <boost/assign.hpp>

using std::cout;
using std::endl;

OpenCLAllKernels::OpenCLAllKernels(CL& cl, fs::path const& source_path, bool debug, bool fast_math)
    : cl_(cl)
{
    typedef std::map<std::string, std::vector<std::string> > Map;
    Map file_to_kernel_names;

    file_to_kernel_names["Images.cl"] = boost::assign::list_of
            ("extractYFloat")("extractCbFloat")("extractCrFloat")("extractYCrCbFloat")
            ("splitFloat3")("mergeFloat3")
            ("convolutionFloatHorizontal")("convolutionFloatVertical")
            ("halfSizeImage")("halfSizeFloat4")("halfSizeFloat4Mean")
            .convert_to_container<std::vector<std::string> > () ;

    file_to_kernel_names["Normals.cl"] = boost::assign::list_of
            ("computeNormals")
            ("smoothNormals")
            .convert_to_container<std::vector<std::string> > () ;

    file_to_kernel_names["Optimize.cl"] = boost::assign::list_of
            ("computeErrorAndGradientReduced")
            ("reduceErrorAndGradient")
            .convert_to_container<std::vector<std::string> > () ;

    file_to_kernel_names["TSDF.cl"] = boost::assign::list_of
            ("addFrame")
            ("addVolume")
            ("setVolumeToSphere")
            ("setMaxWeightInVolume")
            ("setValueInSphere")
            ("setValueInBox")
            ("setPointsInsideBoxTrue")
            ("doesBoxContainSurface")
            (KernelSetFloat::kernel_name.c_str())
            (KernelSetUChar::kernel_name.c_str())
            (KernelSetInt::kernel_name.c_str())
            (KernelSetFloat4::kernel_name.c_str())
            (KernelAddFrameToHistogram::kernel_name.c_str())
            (KernelHistogramSum::kernel_name.c_str())
            (KernelHistogramSumCheckIndex::kernel_name.c_str())
            (KernelHistogramMax::kernel_name.c_str())
            (KernelHistogramMaxCheckIndex::kernel_name.c_str())
            (KernelHistogramVariance::kernel_name.c_str())
            (KernelDivideFloats::kernel_name.c_str())
            (KernelExtractVolumeSlice::kernel_name.c_str())
            (KernelExtractVolumeSliceFloat4::kernel_name.c_str())
            (KernelExtractVolumeSliceFloat4Length::kernel_name.c_str())
            (KernelExtractVolumeFloat::kernel_name.c_str())
            (KernelAddFrame::kernel_name.c_str())
            (KernelAddFrameTo2Means::kernel_name.c_str())
            (KernelAddFrameTo2MeansUsingNormals::kernel_name.c_str())
            (KernelAddFrameTo2MeansUsingStoredNormals::kernel_name.c_str())
            (KernelAddFrameIfCompatible::kernel_name.c_str())
            (KernelRenderPointsAndNormals::kernel_name.c_str())
            (KernelRenderPoints::kernel_name.c_str())
            (KernelRenderNormalForPoints::kernel_name.c_str())
            (KernelRenderColorForPoints::kernel_name.c_str())
            (KernelRender2MeansAbs::kernel_name.c_str())
            (KernelNormalsToShadedImage::kernel_name.c_str())
            (KernelNormalsToColorImage::kernel_name.c_str())
            (KernelPointsToDepthImage::kernel_name.c_str())
            (KernelAddFloats::kernel_name.c_str())
            (KernelAddFloatsWithWeights::kernel_name.c_str())
            (KernelAddFloatsWithWeightsExternalWeight::kernel_name.c_str())
            (KernelGaussianPDF::kernel_name.c_str())
            (KernelGaussianPDFConstantX::kernel_name.c_str())
            (KernelDotVolumeNormal::kernel_name.c_str())
            (KernelMinAbsVolume::kernel_name.c_str())
            (KernelAddVolumes::kernel_name.c_str())
            (KernelBetterNormal::kernel_name.c_str())
            (KernelExtractFloat4ForPointImage::kernel_name.c_str())
            (KernelExtractIntForPointImage::kernel_name.c_str())
            (KernelApplyPoseToNormals::kernel_name.c_str())
            (KernelApplyPoseToPoints::kernel_name.c_str())
            (KernelMinAbsFloatsWithWeights::kernel_name.c_str())
            (KernelMinAbsFloatsWithWeightsRecordIndex::kernel_name.c_str())
            (KernelMinAbsFloatsWithWeightsAndMinimumWeightFraction::kernel_name.c_str())
            (KernelComputeNormalVolume::kernel_name.c_str())
            (KernelComputeNormalVolumeWithWeights::kernel_name.c_str())
            (KernelComputeNormalVolumeWithWeightsUnnormalized::kernel_name.c_str())
            (KernelMaxFloats::kernel_name.c_str())
            (KernelMinFloats::kernel_name.c_str())
            (KernelPickIfIndexFloats::kernel_name.c_str())
            (KernelPickIfIndexFloat4::kernel_name.c_str())
            (KernelSetVolumeSDFBox::kernel_name.c_str())
            (KernelRaytraceBox::kernel_name.c_str())
            (KernelRaytraceSpecial::kernel_name.c_str())
            (KernelMarkPointsViolateEmpty::kernel_name.c_str())
            (KernelDepthImageToPoints::kernel_name.c_str())
            (KernelTransformPoints::kernel_name.c_str())
            (KernelSetInvalidPointsTrue::kernel_name.c_str())
            (KernelOptimizeErrorAndJacobianICP::kernel_name.c_str())
            (KernelOptimizeErrorAndJacobianImage::kernel_name.c_str())
            (KernelOptimizeNormalEquationTerms::kernel_name.c_str())
            (KernelVignetteApplyModelPolynomial3Uchar4::kernel_name.c_str())
            (KernelVignetteApplyModelPolynomial3Float::kernel_name.c_str())

            .convert_to_container<std::vector<std::string> > () ;

    for (Map::const_iterator iter = file_to_kernel_names.begin(); iter != file_to_kernel_names.end(); ++iter) {
        OpenCLKernelsBuilder builder(cl, source_path / iter->first, iter->second, debug, fast_math);
        kernel_name_map_.insert(builder.getKernelMap().begin(), builder.getKernelMap().end());
    }
}


cl::Kernel OpenCLAllKernels::getKernel(std::string const& name) const
{
    std::map<std::string, cl::Kernel>::const_iterator find_iter = kernel_name_map_.find(name);
    if (find_iter == kernel_name_map_.end()) {
        cout << "Unknown kernel: " << name << endl;
        exit(1);
    }
    return find_iter->second;
}

