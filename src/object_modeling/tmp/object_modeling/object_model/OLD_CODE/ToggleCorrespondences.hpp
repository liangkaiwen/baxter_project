#pragma once

#include "ToggleBase.hpp"

template<typename PointT>
class ToggleCorrespondences : public ToggleBase<PointT>
{
protected:
	typename pcl::PointCloud<PointT>::ConstPtr cloud1_;
	typename pcl::PointCloud<PointT>::ConstPtr cloud2_;

public:
	ToggleCorrespondences(std::string name, bool initial_state)
		: ToggleBase<PointT>(name, initial_state) {}

	void setCorrespondences(typename pcl::PointCloud<PointT>::ConstPtr cloud1, typename pcl::PointCloud<PointT>::ConstPtr cloud2) {
		boost::mutex::scoped_lock lock(mtx_);
		if (cloud1->size() != cloud2->size()) throw new exception("cloud1->size() != cloud2->size()");
		cloud1_ = cloud1;
		cloud2_ = cloud2;
		updated_ = true;
	}
	
	virtual void update(pcl::visualization::PCLVisualizer& viewer) {
		// hold lock a short time so that slow slow changes to viewer don't hold up other functions
		bool do_update = false;
		typename pcl::PointCloud<PointT>::ConstPtr safe_cloud1_ptr;
		typename pcl::PointCloud<PointT>::ConstPtr safe_cloud2_ptr;
		{
			boost::mutex::scoped_lock lock(mtx_);
			do_update = updated_;
			safe_cloud1_ptr = cloud1_;
			safe_cloud2_ptr = cloud2_;
			if (do_update) updated_ = false;
		}

		if (do_update && safe_cloud1_ptr && safe_cloud2_ptr) {
			viewer.removeCorrespondences(name_);
			if (state_) {
				// create vector of "matches"
				std::vector<int> matches;
				for (int i = 0; i < safe_cloud1_ptr->size(); ++i) {
					matches.push_back(i);
				}
				viewer.addCorrespondences<PointT>(safe_cloud1_ptr, safe_cloud2_ptr, matches, name_);
			}
		}
	}
};

