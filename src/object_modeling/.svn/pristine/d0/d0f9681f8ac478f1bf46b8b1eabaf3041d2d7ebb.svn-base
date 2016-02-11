#include "volume_modeler_glfw.h"

#include <ctype.h>

#include <iostream>
using std::cout;
using std::endl;

VolumeModelerGLFW * VolumeModelerGLFW::instance_ = NULL;

VolumeModelerGLFW::VolumeModelerGLFW(int width, int height, std::set<std::string> disabled_keys)
	: camera_focus_(0,0,0),
	camera_rotation_(1, 0, 0, 0),
	camera_distance_(1),
	button_left_(false),
	button_middle_(false),
	button_right_(false),
	mouse_x_(0),
	mouse_y_(0),
	last_key_(0),
	clear_color_(0,0,0),
	clear_intensity_(0.5),
	polygon_fill_(true),
	do_destroy_(false),
	window_(NULL),
	point_size_(1),
	enable_read_buffer_(false),
	initial_width_(width),
	initial_height_(height),
	glu_look_at_enabled_(false),
	bipartite_graph_vertex_first_(-1)
{
	// singleton
	instance_ = this;

	// initially disabled keys from set
	typedef std::set<std::string> Set;
	for (Set::const_iterator iter = disabled_keys.begin(); iter != disabled_keys.end(); ++iter) {
		name_to_visibility_[*iter] = false;
	}

	glfwSetErrorCallback(error_callback);
}

void VolumeModelerGLFW::error_callback(int error, const char* description)
{
	cout << description << endl;
}

void VolumeModelerGLFW::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action == GLFW_PRESS) {
		boost::mutex::scoped_lock lock(instance_->mutex_); // needed... key callback in render thread...

		// set for others
		instance_->last_key_ = key;

		if (key == GLFW_KEY_ESCAPE) {
			glfwSetWindowShouldClose(window, GL_TRUE);
		}
		else if (key == GLFW_KEY_L) {
			instance_->polygon_fill_ = !instance_->polygon_fill_;
		}
		else if (key == GLFW_KEY_UP) {
			instance_->point_size_ += 1;
		}
		else if (key == GLFW_KEY_DOWN) {
			instance_->point_size_ -= 1;
			if (instance_->point_size_ < 1) instance_->point_size_ = 1;
		}
		else if (key == GLFW_KEY_LEFT) {
			instance_->clear_intensity_ -= 0.1;
			if (instance_->clear_intensity_ < 0) instance_->clear_intensity_ = 0;
			instance_->clear_color_ = Eigen::Array3f(instance_->clear_intensity_, instance_->clear_intensity_, instance_->clear_intensity_);
		}
		else if (key == GLFW_KEY_RIGHT) {
			instance_->clear_intensity_ += 0.1;
			if (instance_->clear_intensity_ > 1) instance_->clear_intensity_ = 1;
			instance_->clear_color_ = Eigen::Array3f(instance_->clear_intensity_, instance_->clear_intensity_, instance_->clear_intensity_);
		}
		else if (key == GLFW_KEY_J) {
			if (instance_->bipartite_graph_vertex_first_ < 0) instance_->bipartite_graph_vertex_first_ = 0;
			else instance_->bipartite_graph_vertex_first_++;
		}
		else if (key == GLFW_KEY_K) {
			if (instance_->bipartite_graph_vertex_first_ >= 0) instance_->bipartite_graph_vertex_first_--;
		}

		else {
			char key_char = (char)key;
			std::string name(&key_char,1);
			// why am I wasting my time getting this right?
			if (instance_->name_to_visibility_.find(name) == instance_->name_to_visibility_.end()) {
				instance_->name_to_visibility_[name] = false; // assume first press is to turn off?
			}
			else {
				instance_->name_to_visibility_[name] = !instance_->name_to_visibility_[name];
			}
		}
	}
}

int VolumeModelerGLFW::getKeySync()
{
	boost::mutex::scoped_lock lock(mutex_);
	return last_key_;
}

int VolumeModelerGLFW::getKeyLowerCaseSync()
{
	boost::mutex::scoped_lock lock(mutex_);
	return std::tolower(last_key_);
}

void VolumeModelerGLFW::clearKeySync()
{
	boost::mutex::scoped_lock lock(mutex_);
	last_key_ = -1;
}

void VolumeModelerGLFW::button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		instance_->button_left_ = (action == GLFW_PRESS);
	}
	if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
		instance_->button_middle_ = (action == GLFW_PRESS);
	}
	if (button == GLFW_MOUSE_BUTTON_RIGHT) {
		instance_->button_right_ = (action == GLFW_PRESS);
	}
}

float VolumeModelerGLFW::getZFromNormPoint(float x_norm, float y_norm)
{
    const static float trackball_radius = 1.0f;
    float r_squared = trackball_radius * trackball_radius;
    float d_squared = x_norm * x_norm + y_norm * y_norm;
    float z_norm = d_squared > r_squared / 2 ?
                r_squared / 2 / sqrt(d_squared) :
                sqrt(r_squared - d_squared);
    return z_norm;
}

void VolumeModelerGLFW::cursor_callback(GLFWwindow* window, double x, double y)
{
	//cout << x << "," << y << endl;
	float last_x = instance_->mouse_x_;
	float last_y = instance_->mouse_y_;
	instance_->mouse_x_ = x;
	instance_->mouse_y_ = y;

	// need window to normalize
	int width, height;
	glfwGetFramebufferSize(instance_->window_, &width, &height);
	float last_x_norm = (last_x - (width/2)) / (float)(width/2);
	float last_y_norm = (last_y - (height/2)) / (float)(height/2);
	float x_norm = (x - (width/2)) / (float)(width/2);
	float y_norm = (y - (height/2)) / (float)(height/2);
	//cout << x_norm << "," << y_norm << endl;

	const static float translation_factor = 0.5;

	if (instance_->button_left_) {
        // try my own trackball instead...
        float last_z_norm = getZFromNormPoint(last_x_norm, last_y_norm);
        float z_norm = getZFromNormPoint(x_norm, y_norm);
        Eigen::Vector3f v1(last_x_norm, last_y_norm, last_z_norm);
        Eigen::Vector3f v2(x_norm, y_norm, z_norm);

        Eigen::Quaternionf Q;
        Q.setFromTwoVectors(v1,v2);
        // this inverse doesn't really make sense to me, but oh well...
        instance_->camera_rotation_ = Q.inverse() * instance_->camera_rotation_;
        //instance_->camera_rotation_ = instance_->camera_rotation_ * Q.inverse();
        instance_->camera_rotation_.normalize();
	}

	if (instance_->button_right_) {
		float delta_x_norm = x_norm - last_x_norm;
		float delta_y_norm = y_norm - last_y_norm;

		// apply rotation to x and y vectors
		Eigen::Vector3f x_vec(1,0,0);
		Eigen::Vector3f y_vec(0,1,0);
		Eigen::Affine3f apply_to_transation_vectors;
		apply_to_transation_vectors = instance_->camera_rotation_.inverse();
		x_vec = apply_to_transation_vectors * -x_vec;
		y_vec = apply_to_transation_vectors * -y_vec;

        float zoom_based_translation_factor = instance_->camera_distance_ * 0.5;

		instance_->camera_focus_ += x_vec * delta_x_norm * translation_factor * zoom_based_translation_factor; 
		instance_->camera_focus_ += y_vec * delta_y_norm * translation_factor * zoom_based_translation_factor; 
	}

	if (instance_->button_middle_) {
		//float delta_x_norm = x_norm - last_x_norm;
		float delta_y_norm = y_norm - last_y_norm;

		// apply rotation to x and y vectors
		Eigen::Vector3f z_vec(0,0,1);
		Eigen::Affine3f apply_to_transation_vectors;
		apply_to_transation_vectors = instance_->camera_rotation_.inverse();
		z_vec = apply_to_transation_vectors * -z_vec;

		instance_->camera_focus_ += z_vec * delta_y_norm * translation_factor; 
	}
}

void VolumeModelerGLFW::wheel_callback(GLFWwindow* window, double x, double y)
{
	//cout << "wheel: " << x << "," << y << endl;
	const static float factor = 0.1;
    float distance_based_factor = instance_->camera_distance_ * 0.2;
	instance_->camera_distance_ += factor * y * distance_based_factor;
	const static float min_distance = 0.1;
	instance_->camera_distance_ = std::max(instance_->camera_distance_, min_distance);
}

void VolumeModelerGLFW::init()
{
	// currently just one window on creation
	if (!glfwInit())
		exit(EXIT_FAILURE);

	window_ = glfwCreateWindow(initial_width_, initial_height_, "Volume Modeler GLFW", NULL, NULL);
	if (!window_)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window_);

	glfwSetKeyCallback(window_, key_callback);
	glfwSetMouseButtonCallback(window_, button_callback);
	glfwSetCursorPosCallback(window_, cursor_callback);
	glfwSetScrollCallback(window_, wheel_callback);
}

void VolumeModelerGLFW::run()
{
	init();

	const static int sleep = 1000 / 20; // 20 fps max

	// render loop:
	while (!glfwWindowShouldClose(window_))
	{
        if (do_destroy_) break; // new

		render();
		glfwSwapBuffers(window_);
		glfwPollEvents();
		//glfwWaitEvents();
		if (sleep > 0) boost::this_thread::sleep(boost::posix_time::milliseconds(sleep));
	}

	destroy();
}

void VolumeModelerGLFW::runInThread()
{
	if (thread_ptr_) return;

	thread_ptr_.reset(new boost::thread(boost::bind(&VolumeModelerGLFW::run, this)));
}

void VolumeModelerGLFW::destroy()
{
	do_destroy_ = true;
    join(); // new
	if (window_) {
		glfwDestroyWindow(window_);
		window_ = NULL;
	}
	glfwTerminate();
}

void VolumeModelerGLFW::join()
{
    if (thread_ptr_ && boost::this_thread::get_id() != thread_ptr_->get_id()) thread_ptr_->join();
}

void VolumeModelerGLFW::render()
{
	renderInit();
#if 0
	renderAxes();
#endif
#if 0
	Eigen::Affine3f camera_focus_pose;
	camera_focus_pose = Eigen::Translation3f(camera_focus_);
	renderAxes(camera_focus_pose, 0.1);
#endif
	if (polygon_fill_) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
	else {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	}

	renderUpdateInterfaceObjects();
	updateLastReadBuffer();
}

void VolumeModelerGLFW::updateLastReadBuffer()
{
	// a bit sketchy to mess with enable_read_buffer_ while not locked, but whatever
	if (enable_read_buffer_) {
		boost::mutex::scoped_lock lock(mutex_);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glReadBuffer(GL_BACK);
		int width, height;
		glfwGetFramebufferSize(window_, &width, &height);
		last_read_buffer_ = cv::Mat(height, width, CV_8UC3);
		glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, last_read_buffer_.data);
	}
}


void VolumeModelerGLFW::renderInit()
{
	// always?
	glEnable(GL_DEPTH_TEST);

	// size and clear
	int width, height;
	glfwGetFramebufferSize(window_, &width, &height);
	glViewport(0, 0, width, height);
	glClearColor(clear_color_[0], clear_color_[1], clear_color_[2], 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// light every time?
	if (false) {
		glShadeModel(GL_SMOOTH);
		glEnable(GL_COLOR_MATERIAL);
		glEnable(GL_LIGHTING);
		
		const static GLfloat light_position[4] = {10,10,-10,1};
		glLightfv(GL_LIGHT0, GL_POSITION, light_position);
		glEnable(GL_LIGHT0);
	}

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
    gluPerspective(60.0, (float)width / height, 0.01, 1000.);

	// ready to render
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// currently ignore manual camera once update_view_pose ever set
	// need lock to look at update_view_pose_
	{
		boost::mutex::scoped_lock lock(mutex_);
		if (glu_look_at_enabled_) {
			gluLookAt(glu_look_at_position_[0], glu_look_at_position_[1], glu_look_at_position_[2],
				glu_look_at_target_[0], glu_look_at_target_[1], glu_look_at_target_[2],
				glu_look_at_up_[0], glu_look_at_up_[1], glu_look_at_up_[2]);
		}
		else if (update_view_pose_) {
			gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0); // stupid default camera (z forward, -y up)

			Eigen::Affine3f model_pose = update_view_pose_->inverse();
			Eigen::Matrix4f eigen_44 = model_pose.matrix();
			float view_gl[16];
			for (int row = 0; row < 4; ++row) {
				for (int col = 0; col < 4; ++col) {
					view_gl[col * 4 + row] = eigen_44(row,col);
				}
			}
			glMultMatrixf(view_gl);
		}
		else {
			gluLookAt(0,0,-camera_distance_, 0,0,0, 0,-1,0);
	
            // silly rotation...
            {
                Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
                // note negation here...
                m.block<3,3>(0,0) = camera_rotation_.toRotationMatrix();

                float rot_gl[16];
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        rot_gl[col * 4 + row] = m(row,col);
                    }
                }
                glMultMatrixf(rot_gl);
            }

			// first "center" the model
			Eigen::Vector3f translate_camera_focus(-camera_focus_);
			glTranslatef(translate_camera_focus.x(), translate_camera_focus.y(), translate_camera_focus.z());
		}
	}
}

void VolumeModelerGLFW::renderUpdateInterfaceObjects()
{
	boost::mutex::scoped_lock lock(mutex_);

	for (NameToMeshT::iterator iter = name_to_mesh_.begin(); iter != name_to_mesh_.end(); ++iter) {
		if (!name_to_visibility_[iter->first]) continue;
		renderMesh(iter->second);
	}

	for (NameToVerticesT::iterator iter = name_to_lines_.begin(); iter != name_to_lines_.end(); ++iter) {
		if (!name_to_visibility_[iter->first]) continue;
        float scale = getScale(iter->first);
        renderLines(iter->second, scale);
	}

	for (NameToVerticesT::iterator iter = name_to_point_cloud_.begin(); iter != name_to_point_cloud_.end(); ++iter) {
		if (!name_to_visibility_[iter->first]) continue;
		renderPoints(iter->second);
	}

	for (NameToPoseT::iterator iter = name_to_cameras_.begin(); iter != name_to_cameras_.end(); ++iter) {
		if (!name_to_visibility_[iter->first]) continue;
        float scale = getScale(iter->first);
		bool connect_with_lines = false;
		Eigen::Array4ub color;
		bool got_color = getColor(iter->first, color);
		if (got_color) {
			renderPosesCustomColor(iter->second, scale, connect_with_lines, color);
		}
		else {
			renderPoses(iter->second, scale, connect_with_lines);
		}
	}

	for (NameToGraphT::iterator iter = name_to_graph_.begin(); iter != name_to_graph_.end(); ++iter) {
		if (!name_to_visibility_[iter->first]) continue;
		// here do smart stuff with partial selection...
		renderGraph(iter->second.first, iter->second.second);
	}

	for (NameToBipartiteGraphT::iterator iter = name_to_bipartite_graph_.begin(); iter != name_to_bipartite_graph_.end(); ++iter) {
		if (!name_to_visibility_[iter->first]) continue;
		if (bipartite_graph_vertex_first_ >= 0) {
			std::vector<bool> vertices_first_active(iter->second.vertices_first->size(), false);
			if (bipartite_graph_vertex_first_ < vertices_first_active.size()) vertices_first_active[bipartite_graph_vertex_first_] = true;
			renderBipartiteGraph(iter->second.vertices_first, iter->second.vertices_second, iter->second.edges, vertices_first_active);
		}
		else {
			renderBipartiteGraph(iter->second.vertices_first, iter->second.vertices_second, iter->second.edges);
		}
	}

	// alpha at the end
	// not yet implemented
	for (NameToMeshT::iterator iter = name_to_mesh_.begin(); iter != name_to_mesh_.end(); ++iter) {
		//if (!name_to_visibility_[iter->first]) continue;
		//renderAlphaMesh(iter->second);
	}
}

void VolumeModelerGLFW::renderAxes()
{
	glBegin(GL_LINES);
	glColor3ub(255,0,0);
	glVertex3f(0,0,0);
	glVertex3f(1,0,0);

	glColor3ub(0,255,0);
	glVertex3f(0,0,0);
	glVertex3f(0,1,0);

	glColor3ub(0,0,255);
	glVertex3f(0,0,0);
	glVertex3f(0,0,1);

	glEnd();
}

void VolumeModelerGLFW::renderAxes(Eigen::Affine3f const& pose, float line_length, float line_width)
{
	Eigen::Vector3f center(0,0,0);
	Eigen::Vector3f x_axis(line_length,0,0);
	Eigen::Vector3f y_axis(0,line_length,0);
	Eigen::Vector3f z_axis(0,0,line_length);

	center = pose * center;
	x_axis = pose * x_axis;
	y_axis = pose * y_axis;
	z_axis = pose * z_axis;

	glLineWidth(line_width);

	glBegin(GL_LINES);
	glColor3ub(255,0,0);
	glVertex3f(center.x(), center.y(), center.z());
	glVertex3f(x_axis.x(), x_axis.y(), x_axis.z());

	glColor3ub(0,255,0);
	glVertex3f(center.x(), center.y(), center.z());
	glVertex3f(y_axis.x(), y_axis.y(), y_axis.z());

	glColor3ub(0,0,255);
	glVertex3f(center.x(), center.y(), center.z());
	glVertex3f(z_axis.x(), z_axis.y(), z_axis.z());

	glEnd();

	glLineWidth(1);
}

void VolumeModelerGLFW::renderAxesCustomColor(Eigen::Affine3f const& pose, float line_length, float line_width, const Eigen::Array4ub & color)
{
	Eigen::Vector3f center(0,0,0);
	Eigen::Vector3f x_axis(line_length,0,0);
	Eigen::Vector3f y_axis(0,line_length,0);
	Eigen::Vector3f z_axis(0,0,line_length * 2); // double length for z when custom color?

	center = pose * center;
	x_axis = pose * x_axis;
	y_axis = pose * y_axis;
	z_axis = pose * z_axis;

	glLineWidth(line_width);

	glBegin(GL_LINES);
	glColor3ub(color[2], color[1], color[0]); // BGR !!

	glVertex3f(center.x(), center.y(), center.z());
	glVertex3f(x_axis.x(), x_axis.y(), x_axis.z());

	glVertex3f(center.x(), center.y(), center.z());
	glVertex3f(y_axis.x(), y_axis.y(), y_axis.z());

	glVertex3f(center.x(), center.y(), center.z());
	glVertex3f(z_axis.x(), z_axis.y(), z_axis.z());

	glEnd();

	glLineWidth(1);
}

void VolumeModelerGLFW::renderMesh(MeshPtr mesh_ptr)
{
	if (!mesh_ptr) return;

	glBegin(GL_TRIANGLES);

	for (int i = 0; i < mesh_ptr->triangles.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			MeshVertex & v = mesh_ptr->vertices[mesh_ptr->triangles[i][j]];
			glColor3ub(v.c[2], v.c[1], v.c[0]);
			glNormal3f(v.n.x(), v.n.y(), v.n.z());
			glVertex3f(v.p.x(), v.p.y(), v.p.z());
		}
	}

	glEnd();
}

void VolumeModelerGLFW::renderLines(MeshVertexVectorPtr vertices_ptr, float scale)
{
	if (!vertices_ptr) return;

    glLineWidth(scale);

	glBegin(GL_LINES);

	for (int i = 0; i < vertices_ptr->size(); i+=2) {
		MeshVertex & v1 = (*vertices_ptr)[i];
		MeshVertex & v2 = (*vertices_ptr)[i+1];
		glColor3ub(v1.c[2], v1.c[1], v1.c[0]);
		glVertex3f(v1.p.x(), v1.p.y(), v1.p.z());
		glVertex3f(v2.p.x(), v2.p.y(), v2.p.z());
	}

	glEnd();

    glLineWidth(1);
}

void VolumeModelerGLFW::renderPoints(MeshVertexVectorPtr vertices_ptr)
{
	if (!vertices_ptr) return;

	glPointSize(point_size_);
	glBegin(GL_POINTS);

	for (int i = 0; i < vertices_ptr->size(); ++i) {
		MeshVertex & v= (*vertices_ptr)[i];
		glColor3ub(v.c[2], v.c[1], v.c[0]);
		glVertex3f(v.p.x(), v.p.y(), v.p.z());
	}

	glEnd();
}

void VolumeModelerGLFW::renderPoses(PoseListPtrT pose_list_ptr, float scale, bool connect_with_lines)
{
	if (!pose_list_ptr) return;

	for (int i = 0; i < pose_list_ptr->size(); ++i) {
		renderAxes((*pose_list_ptr)[i], 0.1, scale);
		if (connect_with_lines) {
			if (i > 0) {
				Eigen::Vector3f p1 = pose_list_ptr->at(i-1).translation();
				Eigen::Vector3f p2 = pose_list_ptr->at(i).translation();
				glBegin(GL_LINES);
				glColor3ub(255,255,255);
				glVertex3f(p1.x(), p1.y(), p1.z());
				glVertex3f(p2.x(), p2.y(), p2.z());
				glEnd();
			}
		}
	}
}

void VolumeModelerGLFW::renderPosesCustomColor(PoseListPtrT pose_list_ptr, float scale, bool connect_with_lines, const Eigen::Array4ub & color)
{
	if (!pose_list_ptr) return;

	for (int i = 0; i < pose_list_ptr->size(); ++i) {
		renderAxesCustomColor((*pose_list_ptr)[i], 0.1, scale, color);
		if (connect_with_lines) {
			if (i > 0) {
				Eigen::Vector3f p1 = pose_list_ptr->at(i-1).translation();
				Eigen::Vector3f p2 = pose_list_ptr->at(i).translation();
				glBegin(GL_LINES);
				glColor3ub(255,255,255);
				glVertex3f(p1.x(), p1.y(), p1.z());
				glVertex3f(p2.x(), p2.y(), p2.z());
				glEnd();
			}
		}
	}
}

void VolumeModelerGLFW::renderGraph(PoseListPtrT vertices, EdgeListPtrT edges)
{
	std::vector<bool> active_vertices(vertices->size(), true);
	renderGraph(vertices, edges, active_vertices);
}

void VolumeModelerGLFW::renderGraph(PoseListPtrT vertices, EdgeListPtrT edges, std::vector<bool> const& active_vertices)
{
	if (!vertices || !edges) return;

	for (int e = 0; e < edges->size(); ++e) {
		// if either end of edge is active, do it
		int v1 = (*edges)[e].first;
		int v2 = (*edges)[e].second;
		if (!active_vertices[v1] && !active_vertices[v2]) continue;

		// do it
		Eigen::Vector3f p1 = vertices->at(v1).translation();
		Eigen::Vector3f p2 = vertices->at(v2).translation();

		glBegin(GL_LINES);
		glColor3ub(255,255,255);
		glVertex3f(p1.x(), p1.y(), p1.z());
		glVertex3f(p2.x(), p2.y(), p2.z());
		glEnd();
	}
}

void VolumeModelerGLFW::renderBipartiteGraph(PoseListPtrT vertices_first, PoseListPtrT vertices_second, EdgeListPtrT edges)
{
	std::vector<bool> active_vertices_first(vertices_first->size(), true);
	renderBipartiteGraph(vertices_first, vertices_second, edges, active_vertices_first);
}

void VolumeModelerGLFW::renderBipartiteGraph(PoseListPtrT vertices_first, PoseListPtrT vertices_second, EdgeListPtrT edges, std::vector<bool> const& active_vertices_first)
{
	if (!vertices_first || !vertices_second || !edges) return;

	for (int e = 0; e < edges->size(); ++e) {
		// if either end of edge is active, do it
		int v1 = (*edges)[e].first;
		int v2 = (*edges)[e].second;
		if (!active_vertices_first[v1]) continue;

		// do it
		Eigen::Vector3f p1 = vertices_first->at(v1).translation();
		Eigen::Vector3f p2 = vertices_second->at(v2).translation();

		glBegin(GL_LINES);
		glColor3ub(255,255,255);
		glVertex3f(p1.x(), p1.y(), p1.z());
		glVertex3f(p2.x(), p2.y(), p2.z());
		glEnd();
	}
}

void VolumeModelerGLFW::setClearColorSync(float r, float g, float b)
{
	boost::mutex::scoped_lock lock(mutex_);
	clear_color_ = Eigen::Array3f(r,g,b);
}

void VolumeModelerGLFW::updateMesh(const std::string & name, MeshPtr mesh_ptr)
{
	boost::mutex::scoped_lock lock(mutex_);
	name_to_mesh_[name] = mesh_ptr;
	if (name_to_visibility_.find(name) == name_to_visibility_.end()) name_to_visibility_[name] = true;
}

void VolumeModelerGLFW::updateAlphaMesh(const std::string & name, MeshPtr mesh_ptr)
{
	boost::mutex::scoped_lock lock(mutex_);
	name_to_alpha_mesh_[name] = mesh_ptr;
	if (name_to_visibility_.find(name) == name_to_visibility_.end()) name_to_visibility_[name] = true;
}

void VolumeModelerGLFW::updateLines(const std::string & name, MeshVertexVectorPtr vertices_ptr)
{
	boost::mutex::scoped_lock lock(mutex_);
	name_to_lines_[name] = vertices_ptr;
	if (name_to_visibility_.find(name) == name_to_visibility_.end()) name_to_visibility_[name] = true;
}

void VolumeModelerGLFW::updatePointCloud(const std::string & name, MeshVertexVectorPtr vertices_ptr)
{
	boost::mutex::scoped_lock lock(mutex_);
	name_to_point_cloud_[name] = vertices_ptr;
	if (name_to_visibility_.find(name) == name_to_visibility_.end()) name_to_visibility_[name] = true;

}

void VolumeModelerGLFW::updateCameraList(const std::string & name, PoseListPtrT pose_list_ptr)
{
	boost::mutex::scoped_lock lock(mutex_);
	name_to_cameras_[name] = pose_list_ptr;
	if (name_to_visibility_.find(name) == name_to_visibility_.end()) name_to_visibility_[name] = true;
}

void VolumeModelerGLFW::updateViewPose(Eigen::Affine3f const& pose)
{
	boost::mutex::scoped_lock lock(mutex_);
	update_view_pose_.reset(new Eigen::Affine3f(pose));
}

void VolumeModelerGLFW::updateGraph(const std::string & name, PoseListPtrT vertices, EdgeListPtrT edges)
{
	boost::mutex::scoped_lock lock(mutex_);
	name_to_graph_[name] = std::make_pair(vertices, edges);
	if (name_to_visibility_.find(name) == name_to_visibility_.end()) name_to_visibility_[name] = true;
}

void VolumeModelerGLFW::updateBipartiteGraph(const std::string & name, PoseListPtrT vertices_first, PoseListPtrT vertices_second, EdgeListPtrT edges)
{
	boost::mutex::scoped_lock lock(mutex_);
	name_to_bipartite_graph_[name] = BipartiteGraphT(vertices_first, vertices_second, edges);
	if (name_to_visibility_.find(name) == name_to_visibility_.end()) name_to_visibility_[name] = true;
}

void VolumeModelerGLFW::updateScale(const std::string & name, float scale)
{
    boost::mutex::scoped_lock lock(mutex_);
    name_to_scale_[name] = scale;
}

void VolumeModelerGLFW::updateColor(const std::string & name, const Eigen::Array4ub & color)
{
    boost::mutex::scoped_lock lock(mutex_);
    name_to_color_[name] = color;
}

float VolumeModelerGLFW::getScale(const std::string & name)
{
	float scale = 1.0;
	std::map<std::string, float>::iterator find_iter = name_to_scale_.find(name);
    if (find_iter != name_to_scale_.end()) scale = find_iter->second;
    return scale;
}

bool VolumeModelerGLFW::getColor(const std::string & name, Eigen::Array4ub & result_color)
{
	std::map<std::string, Eigen::Array4ub>::iterator find_iter = name_to_color_.find(name);
    if (find_iter != name_to_color_.end()) {
		result_color = find_iter->second;
		return true;
	}
	return false;
}

void VolumeModelerGLFW::setEnableReadBuffer(bool value)
{
	boost::mutex::scoped_lock lock(mutex_);
	enable_read_buffer_ = value;
}

cv::Mat VolumeModelerGLFW::readBuffer()
{
	boost::mutex::scoped_lock lock(mutex_);
	if (!enable_read_buffer_) {
		cout << "warning: you called readBuffer without enable_read_buffer_" << endl;
	}
	cv::Mat result;
	if (!last_read_buffer_.empty()) {
		cv::cvtColor(last_read_buffer_, result, CV_RGB2BGR);
		cv::flip(result, result, 0);
	}
	return result;
}

void VolumeModelerGLFW::setGluLookAt(Eigen::Vector3f const& position, Eigen::Vector3f const& target, Eigen::Vector3f const& up)
{
	boost::mutex::scoped_lock lock(mutex_);
	glu_look_at_position_ = position;
	glu_look_at_target_ = target;
	glu_look_at_up_ = up;
	glu_look_at_enabled_ = true;
}

void VolumeModelerGLFW::disableGluLookAt()
{
	boost::mutex::scoped_lock lock(mutex_);
	glu_look_at_enabled_ = false;
}

