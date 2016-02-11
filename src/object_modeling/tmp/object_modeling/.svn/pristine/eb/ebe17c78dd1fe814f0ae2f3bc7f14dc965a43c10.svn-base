#pragma once

#include "ToggleBase.hpp"

template<typename PointT>
class ToggleMesh : public ToggleBase<PointT>
{
public:
	typedef boost::shared_ptr<std::vector<pcl::Vertices> > VertexVecPtrT;

protected:
	typename pcl::PointCloud<PointT>::ConstPtr cloud_;
	VertexVecPtrT vertex_vec_;
	float alpha_;

public:
	ToggleMesh(std::string name, bool initial_state)
		: ToggleBase<PointT>(name, initial_state)
	{}

	void setCloud(typename pcl::PointCloud<PointT>::ConstPtr cloud, VertexVecPtrT vertex_vec_ptr, float alpha) {
		boost::mutex::scoped_lock lock(this->mtx_);
		this->cloud_ = cloud;
		this->vertex_vec_ = vertex_vec_ptr;
		this->alpha_ = alpha;
		this->updated_ = true;
	}

	virtual void update(pcl::visualization::PCLVisualizer& viewer) {
		boost::mutex::scoped_lock lock(this->mtx_);
		if (this->updated_ && this->cloud_) {
			viewer.removePolygonMesh(this->name_);
			if (this->state_ && !this->cloud_->empty()) {
				pcl::PolygonMesh poly_mesh;
				pcl::toROSMsg(*this->cloud_, poly_mesh.cloud);
				poly_mesh.polygons = *this->vertex_vec_;
				// poly_mesh.header?
				viewer.addPolygonMesh(poly_mesh, this->name_);
				viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, this->alpha_, this->name_);
			}
			this->updated_ = false;
		}
	}
};

