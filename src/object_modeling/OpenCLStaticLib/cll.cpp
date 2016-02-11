#include "stdafx.h"

#include <stdio.h>
#include <string>
#include <iostream>
using std::cout;
using std::endl;

#include "cll.h"
#include "cl_util.h"

CL::CL(OpenCLPlatformType platform_type, OpenCLContextType context_type)
	: platform_type(platform_type),
	context_type(context_type),
	is_initialized(false)
{
	cout << "Initializing OpenCL object and context" << endl;

	const std::string OpenCLPlatformTypeNames[] = {"NVIDIA", "Intel", "AMD"};
	std::string platform_type_name = OpenCLPlatformTypeNames[platform_type];

	std::vector<cl::Platform> platforms;
	err = cl::Platform::get(&platforms);
	printf("cl::Platform::get(): %s\n", oclErrorString(err));
	printf("number of platforms: %d\n", platforms.size());
	if (platforms.size() == 0) {
		printf("Platform size 0\n");
		return;
	}
 
	int platform_index = -1;
	for (int i = 0; i < platforms.size(); i++) {
		std::string platform_name;
		platforms[i].getInfo(CL_PLATFORM_NAME, &platform_name);
		printf("Platform %d: %s\n", i, platform_name.c_str());
		if (platform_name.find(platform_type_name) != std::string::npos) {
			platform_index = i;
		}
	}
	if (platform_index < 0) {
		printf("Could not find desired platform\n");
		return;
	}
	cout << "Found desired platform: " << platform_type_name << endl;

	try {
		cl_context_properties properties[] = 
			{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[platform_index])(), 0};
		if (context_type == OPENCL_CONTEXT_DEFAULT) {
			context = cl::Context(CL_DEVICE_TYPE_DEFAULT, properties);
			cout << "Using CL_DEVICE_TYPE_DEFAULT" << endl;
		}
		else if (context_type == OPENCL_CONTEXT_CPU) {
			context = cl::Context(CL_DEVICE_TYPE_CPU, properties);
			cout << "Using CL_DEVICE_TYPE_CPU" << endl;
		}
		else if (context_type == OPENCL_CONTEXT_GPU) {
			context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
			cout << "Using CL_DEVICE_TYPE_GPU" << endl;
		}
		else {
			cout << "HOW THE HELL DID YOU GET HERE?" << endl;
			return;
		}
		devices = context.getInfo<CL_CONTEXT_DEVICES>();
		printf("Number of devices %d\n", devices.size());
		if (devices.size() == 0) {
			cout << "No devices" << endl;
			return;
		}
		printf("Currently defaulting to device 0\n");
		queue = cl::CommandQueue(context, devices[0], 0, &err);
	}
	catch (cl::Error e) {
		printf("ERROR: %s(%d)\n", e.what(), e.err());
		return;
	}

	is_initialized = true;
}


int CL::loadProgram(std::string program_source, std::string source_full_path, std::string additional_args)
{
	int pl;
	printf("create program\n");
	pl = program_source.size();
	printf("kernel size: %d\n", pl);
	cl::Program program;
	try
	{
		cl::Program::Sources source(1,
			std::make_pair(program_source.c_str(), pl));
		program = cl::Program(context, source);
	}
	catch (cl::Error er) {
		printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
		return -1;
	}

	printf("build program:\n");
	try
	{
		std::string build_options;
		if (!source_full_path.empty()) {
			build_options += " -g -s \"" + source_full_path + "\"";
		}
		build_options += " " + additional_args;
		err = program.build(devices, build_options.c_str());
	}
	catch (cl::Error er) {
		printf("program.build: %s\n", oclErrorString(er.err()));
		std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
		throw std::runtime_error("BUILD ERROR");
		return -1;
	}
	printf("done building program\n");
	std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << std::endl;
	std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]) << std::endl;
	std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;

	programs.push_back(program);
	return ((int) programs.size() - 1);
}

