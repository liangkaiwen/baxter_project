#include "pose_provider_base.h"

PoseProviderBase::PoseProviderBase()
{
}

bool PoseProviderBase::getNextPose(Eigen::Affine3f & result_pose)
{
    return false;
}
