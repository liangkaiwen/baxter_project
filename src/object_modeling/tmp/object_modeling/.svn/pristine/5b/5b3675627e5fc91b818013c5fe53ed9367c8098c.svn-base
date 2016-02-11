#pragma once

#include "pose_provider_base.h"

class PoseProviderStandard : public PoseProviderBase
{
public:
    PoseProviderStandard(fs::path camera_list_txt);

    virtual bool getNextPose(Eigen::Affine3f & result_pose);

protected:
    PosePtrList camera_list_vec_;
    PosePtrList::iterator camera_list_iter_;
};

