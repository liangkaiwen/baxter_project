#include "volume_modeler_glut.h"

#include <iostream>
using std::cout;
using std::endl;

VolumeModelerGlut * VolumeModelerGlut::instance = NULL;

VolumeModelerGlut::VolumeModelerGlut(int argc, char* argv[])
	: near_plane_(0.01),
	far_plane_(100),
	window_width_(800),
	window_height_(600),
	button_1_down_(false)
{
	// this is a singleton class (not enforced, of course!)
	VolumeModelerGlut::instance = this;

	// GLUT Window Initialization:
	glutInit (&argc, argv);
	glutInitWindowSize (window_width_, window_height_);
	glutInitDisplayMode ( GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutCreateWindow ("Volume Modeler GLUT");

	// graphics mode:
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	// Register callbacks:
	glutDisplayFunc (&VolumeModelerGlut::display);
	glutReshapeFunc (&VolumeModelerGlut::reshape);
	glutMouseFunc (&VolumeModelerGlut::mouseButton);
	glutMotionFunc (&VolumeModelerGlut::mouseMotion);
	//glutKeyboardFunc (Keyboard);
	//glutIdleFunc (AnimateScene);
}


void VolumeModelerGlut::renderScene()
{

}

void VolumeModelerGlut::display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	gluLookAt(0, 0, -5, 0, 0, -1, 0, 1, 0); // stupid default camera
	const static GLfloat light_position[4] = {100,100,100,1};
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	renderScene();
	glutSwapBuffers();
}

void VolumeModelerGlut::reshape(GLint width, GLint height)
{
	instance->window_width_ = width;
	instance->window_height_ = height;
	glViewport(0, 0, instance->window_width_, instance->window_height_);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (float)instance->window_width_ / instance->window_height_, instance->near_plane_, instance->far_plane_);
	glMatrixMode(GL_MODELVIEW);
}

void VolumeModelerGlut::mouseButton(int button, int state, int x, int y)
{
	// Respond to mouse button presses.
	// If button1 pressed, mark this state so we know in motion function.
	if (button == GLUT_LEFT_BUTTON)
	{
		instance->button_1_down_ = (state == GLUT_DOWN);
		cout << "mouseButton: " << x << "," << y << endl;
	}
}

void VolumeModelerGlut::mouseMotion(int x, int y)
{
	if (instance->button_1_down_)
	{
		cout << "mouseMotion: " << x << "," << y << endl;
		glutPostRedisplay();
	}
}