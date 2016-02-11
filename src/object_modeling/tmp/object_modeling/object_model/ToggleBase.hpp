#pragma once

#include <pcl/visualization/pcl_visualizer.h>

template<typename PointT>
class ToggleBase
{
protected:
	boost::mutex mtx_;
	const std::string name_;
	bool state_;
	bool updated_;

public:
	ToggleBase(std::string name, bool initial_state)
		: name_(name),
		state_(initial_state),
		updated_(true)
	{}

	void toggleState() {
		boost::mutex::scoped_lock lock(mtx_);
		state_ = !state_;
		updated_ = true;
	}

	bool getState() {
		boost::mutex::scoped_lock lock(mtx_);
		return state_;
	}

	void setState(bool state) {
		boost::mutex::scoped_lock lock(mtx_);
		state_ = state;
	}
	
	virtual void update(pcl::visualization::PCLVisualizer& viewer) = 0;
};

