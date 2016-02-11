#include "pose_provider_standard.h"

#include "EigenUtilities.h"

PoseProviderStandard::PoseProviderStandard(fs::path camera_list_txt)
{
    EigenUtilities::loadPosesFromFile(camera_list_txt, camera_list_vec_);
    camera_list_iter_ = camera_list_vec_.begin();
}

bool PoseProviderStandard::getNextPose(Eigen::Affine3f & result_pose)
{
    if (camera_list_iter_ != camera_list_vec_.end()) {
        result_pose = **camera_list_iter_;
        camera_list_iter_++;
        return true;
    }
    return false;
}
