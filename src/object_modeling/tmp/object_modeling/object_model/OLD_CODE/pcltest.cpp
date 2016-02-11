// pcltest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>
#include <opencv2\opencv.hpp>

#include <XnOpenNI.h>
#include <XnCppWrapper.h>

#define CAM_WIDTH		640
#define CAM_HEIGHT		480
#define CAM_FPS			30

#define CHECK_RC(rc, what) if (rc != XN_STATUS_OK) { printf("%s failed: %s\n", what, xnGetStatusString(rc)); return rc; }

int _tmain(int argc, _TCHAR* argv[])
{

	xn::Context oniContext;
	xn::DepthGenerator oniDepth;
	xn::ImageGenerator oniImage;
	
	//begin kinect initialize
	XnStatus nRetVal = XN_STATUS_OK; 

	nRetVal = oniContext.Init(); 
	CHECK_RC(nRetVal, "Initialize context"); 
	
	nRetVal = oniDepth.Create(oniContext); 
	CHECK_RC(nRetVal, "Create depth generator");

	nRetVal = oniImage.Create(oniContext); 
	CHECK_RC(nRetVal, "Create image generator");
	oniImage.SetPixelFormat(XN_PIXEL_FORMAT_RGB24);

	XnMapOutputMode mapModeVGA;
	mapModeVGA.nXRes = CAM_WIDTH; 
	mapModeVGA.nYRes = CAM_HEIGHT; 
	mapModeVGA.nFPS = CAM_FPS;

	nRetVal = oniDepth.SetMapOutputMode(mapModeVGA); 
	CHECK_RC(nRetVal, "SetMapOutputMode for depth generator");
	nRetVal = oniImage.SetMapOutputMode(mapModeVGA); 
	CHECK_RC(nRetVal, "SetMapOutputMode for image generator");

	nRetVal = oniContext.StartGeneratingAll(); 
	CHECK_RC(nRetVal, "StartGeneratingAll");

	oniDepth.GetAlternativeViewPointCap().SetViewPoint(oniImage);

	//End kinect initialize

	//get RGB image and depth image
	nRetVal = oniContext.WaitAnyUpdateAll();
	CHECK_RC(nRetVal, "UpdateData");
	const XnRGB24Pixel *pix = oniImage.GetRGB24ImageMap();
	const XnDepthPixel *dpix = oniDepth.GetDepthMap();

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr nearCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	
	// Fill in the cloud data
	nearCloud->width    = CAM_WIDTH;
	nearCloud->height   = CAM_HEIGHT;
	nearCloud->is_dense = true;
	nearCloud->points.resize (nearCloud->width * nearCloud->height);

	pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");

	while (!viewer.wasStopped ())
	{

		nRetVal = oniContext.WaitAnyUpdateAll();
		CHECK_RC(nRetVal, "UpdateData");
		const XnRGB24Pixel *pix = oniImage.GetRGB24ImageMap();
		const XnDepthPixel *dpix = oniDepth.GetDepthMap();

		//GeneratePointCloud(oniDepth, dpix, pix, nearCloud);

		for(int i = 0, idx = 0; i < CAM_HEIGHT; i++)
		{
			for(int j = 0; j < CAM_WIDTH; j++, idx++)
			{
				nearCloud->points[idx].x = j;
				nearCloud->points[idx].y = i;
				if(dpix[idx] == 0 || dpix[idx] > 1200)
				{
					nearCloud->points[idx].x = NULL;
					nearCloud->points[idx].y = NULL;
					nearCloud->points[idx].z = NULL;
				}
				else
					nearCloud->points[idx].z = dpix[idx];

				nearCloud->points[idx].r = dpix[idx];
				nearCloud->points[idx].g = 0;
				nearCloud->points[idx].b = 0;
			}
		}


		viewer.showCloud(nearCloud);
	}

	return (0);
}



