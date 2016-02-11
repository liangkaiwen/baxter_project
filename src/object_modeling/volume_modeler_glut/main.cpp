#include "volume_modeler_glut.h"

#include <boost/thread.hpp>

#include <iostream>
using std::cout;
using std::endl;

void glutForever()
{
	while(true) {
		cout << "about to glutMainLoopEvent..." << endl;
		glutMainLoopEvent();
	}
}

int main(int argc, char** argv)
{
	VolumeModelerGlut(argc, argv);

	//glutMainLoop ();

	//while(true) {
	//	glutMainLoopEvent();
	//}

	boost::thread t (&glutForever);
	cout << "thread launched..." << endl;
	t.join();

	return 0;
}
