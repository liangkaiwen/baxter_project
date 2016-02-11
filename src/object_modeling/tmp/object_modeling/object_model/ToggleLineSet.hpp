#pragma once

#include "ToggleBase.hpp"

template<typename PointT>
class ToggleLineSet : public ToggleBase<PointT>
{
protected:
	typename pcl::PointCloud<PointT>::ConstPtr cloud_;
	int currentLineSetSize_; // for removal by generated names
	vtkSmartPointer<vtkActor> actor_; // for vtk hack

	std::string getLineName(int i) {
		return (this->name_ + boost::lexical_cast<std::string>(i));
	}

	void removeLines(pcl::visualization::PCLVisualizer& viewer) {
		for (int i = 0; i < currentLineSetSize_; i++) {
			viewer.removeShape(getLineName(i));
		}
		currentLineSetSize_ = 0;
	}

	void addLines(pcl::visualization::PCLVisualizer& viewer, typename pcl::PointCloud<PointT>::ConstPtr safe_cloud_ptr) {
		pcl::ScopeTime st ( ("[TIMING] addLines in ToggleLineSet: " + this->name_).c_str());

		for (size_t i = 0; i < safe_cloud_ptr->points.size() / 2; i++) {
			const PointT& p1 = safe_cloud_ptr->points[2*i];
			const PointT& p2 = safe_cloud_ptr->points[2*i + 1];

			float r = (float) p1.r / 255.f;
			float g = (float) p1.g / 255.f;
			float b = (float) p1.b / 255.f;
			viewer.addLine(p1, p2, r, g, b, getLineName(i));
			currentLineSetSize_ = i+1;
		}
	}

	void removeLinesVTK(pcl::visualization::PCLVisualizer& viewer) {
		viewer.getRenderWindow()->GetRenderers()->GetFirstRenderer()->RemoveActor(actor_);
	}

	void addLinesVTK(pcl::visualization::PCLVisualizer& viewer, typename pcl::PointCloud<PointT>::ConstPtr safe_cloud_ptr) {
		vtkSmartPointer<vtkAppendPolyData> line_data_append = vtkSmartPointer<vtkAppendPolyData>::New ();
		vtkSmartPointer<vtkUnsignedCharArray> line_colors = vtkSmartPointer<vtkUnsignedCharArray>::New ();
		line_colors->SetNumberOfComponents (3);
		line_colors->SetName ("Colors");

		for (size_t i = 0; i < safe_cloud_ptr->points.size() / 2; i++) {
			const PointT& p1 = safe_cloud_ptr->points[2*i];
			const PointT& p2 = safe_cloud_ptr->points[2*i + 1];
			unsigned char rgb[3] = {p1.r, p1.g, p1.b};

			vtkSmartPointer<vtkLineSource> line = vtkSmartPointer<vtkLineSource>::New ();
			line->SetPoint1 (p1.x, p1.y, p1.z);
			line->SetPoint2 (p2.x, p2.y, p2.z);

			line_data_append->AddInput(line->GetOutput());
			line_colors->InsertNextTupleValue(rgb);
		}
		line_data_append->Update();
		vtkSmartPointer<vtkPolyData> line_data = line_data_append->GetOutput ();
		line_data->GetCellData ()->SetScalars (line_colors);

		vtkSmartPointer<vtkPolyDataMapper> map = vtkSmartPointer<vtkPolyDataMapper>::New();  // added this line 
		map->SetInput(line_data); // added this line 

		//vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();// added this line 
		actor_ = vtkSmartPointer<vtkActor>::New();// added this line 
		actor_->SetMapper(map);// added this line 

		actor_->GetProperty ()->SetRepresentationToWireframe (); 
		//actor->GetProperty ()->SetOpacity (0.5); 

		viewer.getRenderWindow()->GetRenderers()->GetFirstRenderer()->AddActor(actor_);
	}

public:
	ToggleLineSet(std::string name, bool initial_state)
		: ToggleBase<PointT>(name, initial_state),
		currentLineSetSize_(0)
	{}

	void setLineSet(typename pcl::PointCloud<PointT>::ConstPtr cloud) {
		boost::mutex::scoped_lock lock(this->mtx_);
		if (cloud->points.size() % 2 != 0) throw std::runtime_error("cloud->points.size() % 2 != 0");
		this->cloud_ = cloud;
		this->updated_ = true;
	}
	
	virtual void update(pcl::visualization::PCLVisualizer& viewer) {
		// hold lock a short time so that slow slow changes to viewer don't hold up other functions
		bool do_update = false;
		typename pcl::PointCloud<PointT>::ConstPtr safe_cloud_ptr;
		{
			boost::mutex::scoped_lock lock(this->mtx_);
			do_update = this->updated_;
			safe_cloud_ptr = this->cloud_;
			if (do_update) this->updated_ = false;
		}

		if (do_update && safe_cloud_ptr) {
			//removeLines(viewer);
			//removeLinesAsMesh(viewer);
			removeLinesVTK(viewer);
			if (this->state_ && !safe_cloud_ptr->empty()) {
				//addLines(viewer, safe_cloud_ptr);
				//addLinesAsMesh(viewer, safe_cloud_ptr);
				addLinesVTK(viewer, safe_cloud_ptr);
			}
		}
	}
};

