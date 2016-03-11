/******************************************************************************
	INTEL CORPORATION PROPRIETARY INFORMATION
	This software is supplied under the terms of a license agreement or nondisclosure
	agreement with Intel Corporation and may not be copied or disclosed except in
	accordance with the terms of that agreement
	Copyright(c) 2011-2015 Intel Corporation. All Rights Reserved.
*******************************************************************************/

#include "ros/ros.h"
#include "std_msgs/String.h"
 
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include "std_msgs/Float32MultiArray.h"

#include "realsense_image_viewer_nodelet.h"

//Nodelet dependencies
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(realsense::CImageViewerNodelet, nodelet::Nodelet)

namespace realsense
{


	//******************************
	// Public Methods
	//******************************

	CImageViewerNodelet::~CImageViewerNodelet()
	{
		cv::destroyAllWindows();
	}

	void CImageViewerNodelet::onInit()
	{

		ros::NodeHandle& nh = getNodeHandle();
		cv::namedWindow("viewDepth");
		cv::namedWindow("viewColor");
		cv::startWindowThread();

		image_transport::ImageTransport it(nh);
		m_sub_depth = it.subscribe("camera/depth/image_raw", 1, CImageViewerNodelet::imageDepthCallback);
		m_sub_color = it.subscribe("camera/color/image_raw", 1, CImageViewerNodelet::imageColorCallback);

		ROS_INFO_STREAM("Starting RealSense Image Viewer node");

		return;
	 }

	//******************************
	// Private Methods
	//******************************

	void CImageViewerNodelet::imageDepthCallback(const sensor_msgs::ImageConstPtr& msg)
	{
		// Simply cliping the values so we can view them in gray scale.
		try
		{
			cv::Mat image =  cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
			cv::Mat imageDepth(image.rows,image.cols, CV_8UC1, cv::Scalar(0));

			uint16_t *imageDepthPtr = (uint16_t *)image.data;

			uchar * imagePtr = imageDepth.data;

			uint y,x;

			for (y =0 ; y < image.rows ; y++)
			{
				int pix;

				int MAX = 2000;//2000 for DS4, 1000 for IVCAM
				for (x = 0; x < image.cols; x++)
				{
					pix = *imageDepthPtr++;
					if (pix <= 0)
					{
						pix = MAX;
					}
					else if (pix > MAX)
					{
						pix = MAX;
					}


					pix = (int)( (1 -  ((double)pix / MAX))  *255);

					*imagePtr++ = pix;

				}
			}

			cv::imshow("viewDepth", imageDepth);

		}
		catch (cv_bridge::Exception& e)
		{
			ROS_ERROR("Could not convert from '%s' to 'mono16'.", msg->encoding.c_str());
		}
	}

	void CImageViewerNodelet::imageColorCallback(const sensor_msgs::ImageConstPtr& msg)
	{
		try
		{
			cv::Mat imageColor =  cv_bridge::toCvShare(msg, "bgr8")->image;

			cv::imshow("viewColor", imageColor);
		}
		catch (cv_bridge::Exception& e)
		{
			ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
		}
	}

}
