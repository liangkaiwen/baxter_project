#pragma once

#include <EigenUtilities.h>
#include <opencv2/opencv.hpp>

#include "cll.h"

const char* oclErrorString(cl_int error);

cl_float16 getCLPose(const Eigen::Affine3f& pose);

int getNumBlocks(int inputSize, int numThreads);

void debugFinishPrintError(CL & cl, bool verbose);


