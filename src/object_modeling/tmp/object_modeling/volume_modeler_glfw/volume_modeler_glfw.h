#pragma once

#define GLFW_INCLUDE_GLU 1
#include <GLFW/glfw3.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>

#include <boost/thread.hpp>

#include "update_interface.h"

#include <set>

class VolumeModelerGLFW : public UpdateInterface
{
protected:
	// types
	typedef std::map<std::string, MeshPtr> NameToMeshT;
	typedef std::map<std::string, MeshVertexVectorPtr> NameToVerticesT;
	typedef std::map<std::string, PoseListPtrT> NameToPoseT;
	typedef std::map<std::string, std::pair<PoseListPtrT, EdgeListPtrT> >  NameToGraphT;

	struct BipartiteGraphT {
		PoseListPtrT vertices_first;
		PoseListPtrT vertices_second;
		EdgeListPtrT edges;

		// null default constructor
		BipartiteGraphT()
		{}

		BipartiteGraphT(PoseListPtrT vertices_first, PoseListPtrT vertices_second, EdgeListPtrT edges)
			: vertices_first(vertices_first),
			vertices_second(vertices_second),
			edges(edges)
		{}
	};
	typedef std::map<std::string, BipartiteGraphT> NameToBipartiteGraphT;

public:
	VolumeModelerGLFW(int width, int height, std::set<std::string> disabled_keys = std::set<std::string>());

	static void error_callback(int error, const char* description);
	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void button_callback(GLFWwindow* window, int button, int action, int mods);
	static void cursor_callback(GLFWwindow* window, double x, double y);
	static void wheel_callback(GLFWwindow* window, double x, double y);

    static float getZFromNormPoint(float x_norm, float y_norm);

	void runInThread();
	void join();
	void destroy();

	int getKeySync();
	int getKeyLowerCaseSync();
	void clearKeySync();

	void setClearColorSync(float r, float g, float b);

	void setEnableReadBuffer(bool value);
	cv::Mat readBuffer();

	void setGluLookAt(Eigen::Vector3f const& position, Eigen::Vector3f const& target, Eigen::Vector3f const& up);
	void disableGluLookAt();

	// update interface
	virtual void updateMesh(const std::string & name, MeshPtr mesh_ptr);
	virtual void updateAlphaMesh(const std::string & name, MeshPtr mesh_ptr);
	virtual void updateLines(const std::string & name, MeshVertexVectorPtr vertices_ptr);
	virtual void updatePointCloud(const std::string & name, MeshVertexVectorPtr vertices_ptr);
	virtual void updateCameraList(const std::string & name, PoseListPtrT pose_list_ptr);
	virtual void updateGraph(const std::string & name, PoseListPtrT vertices, EdgeListPtrT edges);
	virtual void updateBipartiteGraph(const std::string & name, PoseListPtrT vertices_first, PoseListPtrT vertices_second, EdgeListPtrT edges);

    virtual void updateScale(const std::string & name, float scale);
	virtual void updateColor(const std::string & name, const Eigen::Array4ub & color); // external assumed to be BGR!!!

    virtual void updateViewPose(Eigen::Affine3f const& pose);

protected:
	// functions
	void run();
	void init();
	void render();

	void updateLastReadBuffer();

	void renderInit();
	void renderUpdateInterfaceObjects();
	void renderAxes();
	void renderAxes(Eigen::Affine3f const& pose, float line_length, float line_width);
	void renderAxesCustomColor(Eigen::Affine3f const& pose, float line_length, float line_width, const Eigen::Array4ub & color);
	void renderMesh(MeshPtr mesh_ptr);
    void renderLines(MeshVertexVectorPtr vertices_ptr, float scale);
	void renderPoints(MeshVertexVectorPtr vertices_ptr);
	void renderPoses(PoseListPtrT pose_list_ptr, float scale, bool connect_with_lines);
	void renderPosesCustomColor(PoseListPtrT pose_list_ptr, float scale, bool connect_with_lines, const Eigen::Array4ub & color);
	void renderGraph(PoseListPtrT vertices, EdgeListPtrT edges);
	void renderGraph(PoseListPtrT vertices, EdgeListPtrT edges, std::vector<bool> const& active_vertices);
	void renderBipartiteGraph(PoseListPtrT vertices_first, PoseListPtrT vertices_second, EdgeListPtrT edges);
	void renderBipartiteGraph(PoseListPtrT vertices_first, PoseListPtrT vertices_second, EdgeListPtrT edges, std::vector<bool> const& active_vertices_first);

    float getScale(const std::string & name);
	bool getColor(const std::string & name, Eigen::Array4ub & result_color);


	// members
	static VolumeModelerGLFW * instance_;

	GLFWwindow* window_;

	bool do_destroy_;

	boost::shared_ptr<boost::thread> thread_ptr_;
	boost::mutex mutex_;

	NameToMeshT name_to_mesh_;
	NameToMeshT name_to_alpha_mesh_;
	NameToVerticesT name_to_lines_;
	NameToVerticesT name_to_point_cloud_;
	NameToPoseT name_to_cameras_;
	NameToGraphT name_to_graph_;
	NameToBipartiteGraphT name_to_bipartite_graph_;

    std::map<std::string, float> name_to_scale_; // generalize for any object.  Some objects might not care
	std::map<std::string, Eigen::Array4ub> name_to_color_; // similar to scale...can make certain types care...

	std::map<std::string, bool> name_to_visibility_;

	// set to not-null to take effect on next render
	boost::shared_ptr<Eigen::Affine3f> update_view_pose_;

	Eigen::Vector3f camera_focus_;
	Eigen::Quaternionf camera_rotation_;
	float camera_distance_;

	bool button_left_;
	bool button_middle_;
	bool button_right_;
	float mouse_x_;
	float mouse_y_;

	int last_key_;

	Eigen::Array3f clear_color_;
	float clear_intensity_; // when modified, will overwrite color

	bool polygon_fill_;

	int point_size_;

	bool enable_read_buffer_;
	cv::Mat last_read_buffer_;

	int initial_width_;
	int initial_height_;

	bool glu_look_at_enabled_;
	Eigen::Vector3f glu_look_at_position_;
	Eigen::Vector3f glu_look_at_target_;
	Eigen::Vector3f glu_look_at_up_;

	int bipartite_graph_vertex_first_;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
