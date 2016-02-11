#pragma once

#include <GL/glut.h>
#include <GL/freeglut_ext.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

class VolumeModelerGlut
{
public:
	VolumeModelerGlut(int argc, char* argv[]);

	static void renderScene();
	static void display();
	static void reshape(GLint width, GLint height);
	static void mouseButton(int button, int state, int x, int y);
	static void mouseMotion(int x, int y);


protected:
	static VolumeModelerGlut * instance;

	GLfloat near_plane_;
	GLfloat far_plane_;
	int window_width_;
	int window_height_;

	bool button_1_down_;



public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};