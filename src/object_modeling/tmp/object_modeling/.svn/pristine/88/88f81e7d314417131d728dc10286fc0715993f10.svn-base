#pragma once

#include "ToggleBase.hpp"

template<typename PointT>
class ToggleNormals : public ToggleBase<PointT>
{
protected:
	typename pcl::PointCloud<PointT>::ConstPtr cloud_;
	int normal_freq_;
	float normal_scale_;

public:
	ToggleNormals(std::string name, bool initial_state, int normal_freq = 1, float normal_scale = 0.01) 
		: ToggleBase<PointT>(name, initial_state),
		normal_freq_(normal_freq),
		normal_scale_(normal_scale)
	{}

	void setCloud(typename pcl::PointCloud<PointT>::ConstPtr cloud) {
		boost::mutex::scoped_lock lock(this->mtx_);
		this->cloud_ = cloud;
		this->updated_ = true;
	}

	virtual void update(pcl::visualization::PCLVisualizer& viewer) {
		boost::mutex::scoped_lock lock(this->mtx_);
		if (this->updated_ && this->cloud_) {
			viewer.removeShape(this->name_);
			if (this->state_) {
				viewer.addPointCloudNormals<PointT> (this->cloud_, this->normal_freq_, this->normal_scale_, this->name_);
			}
			this->updated_ = false;
		}
	}
};

