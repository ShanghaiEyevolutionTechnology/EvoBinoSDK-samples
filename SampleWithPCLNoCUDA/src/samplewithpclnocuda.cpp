/**************************************************************************************************
** This sample demonstrates how to use EvoBinoSDK with PCL without CUDA                          **
***************************************************************************************************/


//standard header
#include <iostream>
#include <string>

//PCL header
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

//SDK header
#include "evo_stereocamera.h"

//Define Point Type
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

int main(int argc, char** argv)
{
	//define camera
	evo::bino::StereoCamera camera;
	//open camera
	evo::bino::RESOLUTION_FPS_MODE res_mode = evo::bino::RESOLUTION_FPS_MODE_HD720_60;
	evo::RESULT_CODE res = camera.open(res_mode);
	std::cout << "camera open: " << result_code2str(res) << std::endl;
	
	if (res == evo::RESULT_CODE_OK)//open camera successed
	{
		//create empty point cloud with smart ptr
		PointCloud::Ptr cloud(new PointCloud);

		//create PCL viewer
		boost::shared_ptr<pcl::visualization::CloudViewer> viewer(new pcl::visualization::CloudViewer("Cloud Viewer"));
		
		//grab parameters
		evo::bino::GrabParameters grab_parameters;
		grab_parameters.do_rectify = true;
		grab_parameters.calc_disparity = true;
		grab_parameters.calc_distance = true;

		//set unit of measurement
		camera.setMeasureUnit(evo::bino::MEASURE_UNIT_METER);

		//evo Mat for point cloud
		evo::Mat<float> evo_pointcloud;//evo Mat for xyz-bgra

		//main loop
		while (!viewer->wasStopped(10))
		{
			// Get frames and launch the computation
			if (camera.grab(grab_parameters) == evo::RESULT_CODE_OK)
			{
				//retrieve point cloud
				evo_pointcloud = camera.retrieveDepth(evo::bino::DEPTH_TYPE_POINT_CLOUD_UNORGANIZED_XYZBGRA);

				//set to PCL pointcloud
				cloud->clear();
				for (int m = 0; m < evo_pointcloud.getWidth() * evo_pointcloud.getHeight(); m++){
					PointT p;
					p.x = evo_pointcloud.data[m * 4];			//X
					p.y = evo_pointcloud.data[m * 4 + 1];		//Y
					p.z = evo_pointcloud.data[m * 4 + 2];		//Z
					int * color_ptr = (int*)evo_pointcloud.data + m * 4 + 3;	//B-G-R-A
					p.rgba = *color_ptr;
					cloud->push_back(p);	//add point to cloud
				}
				//show point cloud
				viewer->showCloud(cloud);
			}
		}

		cloud->points.clear();
	}
	camera.close();

	return 0;
}