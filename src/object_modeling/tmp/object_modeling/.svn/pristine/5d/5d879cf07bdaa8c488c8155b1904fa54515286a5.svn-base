#pragma once

#include "ToggleBase.hpp"

template<typename PointT>
class ToggleCloud : public ToggleBase<PointT>
{
protected:
	typename pcl::PointCloud<PointT>::ConstPtr cloud_;

public:
	ToggleCloud(std::string name, bool initial_state) :
		ToggleBase<PointT>(name, initial_state)
	{}

	void setCloud(typename pcl::PointCloud<PointT>::ConstPtr cloud) {
		boost::mutex::scoped_lock lock(this->mtx_);
		this->cloud_ = cloud;
		this->updated_ = true;
	}

	virtual void update(pcl::visualization::PCLVisualizer& viewer) {
		boost::mutex::scoped_lock lock(this->mtx_);
		if (this->updated_ && this->cloud_) {
			viewer.removePointCloud(this->name_);
			if (this->state_) {
				viewer.addPointCloud(this->cloud_, this->name_);
			}
			this->updated_ = false;
		}
	}
};

