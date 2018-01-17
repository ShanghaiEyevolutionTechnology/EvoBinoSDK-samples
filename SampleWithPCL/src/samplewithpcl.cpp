/**************************************************************************************************
** This sample demonstrates how to use Evo SDK with PCL                                          **
***************************************************************************************************/


//standard header
#include <iostream>
#include <string>
using namespace std;

//PCL header
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

//Evo SDK header
#include "evo_global_define.h"//global define
#include "evo_depthcamera.h"//depth camera

//Define Point Type
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

int main(int argc, char** argv)
{
	//define camera
	evo::bino::DepthCamera camera;
	//open camera
	evo::bino::RESOLUTION_FPS_MODE res_mode = evo::bino::RESOLUTION_FPS_MODE_HD720_60;
	evo::bino::RESULT_CODE res = camera.open(res_mode);
	std::cout << "depth camera open: " << result_code2str(res) << std::endl;
	
	if (res == evo::bino::RESULT_CODE_OK)//open camera successed
	{
		//create empty point cloud with smart ptr
		PointCloud::Ptr cloud(new PointCloud);

		//create PCL viewer
		boost::shared_ptr<pcl::visualization::CloudViewer> viewer(new pcl::visualization::CloudViewer("Cloud Viewer"));
		
		//grab parameters
		evo::bino::GrabParameters grab_parameters;

		//set unit of measurement
		camera.setMeasureUnit(evo::bino::MEASURE_UNIT_METER);

		//evo Mat for point cloud
		evo::Mat<float> evo_pointcloud;//evo Mat for xyz-bgra

		//main loop
		while (!viewer->wasStopped(10))
		{
			// Get frames and launch the computation
			if (camera.grab(grab_parameters) == evo::bino::RESULT_CODE_OK)
			{
				//retrieve point cloud
				evo_pointcloud = camera.retrieveDepth(evo::bino::DEPTH_TYPE_POINT_CLOUD_XYZBGRA, evo::MAT_TYPE_CPU);

				//set to PCL pointcloud
				cloud->clear();
				for (int m = 0; m < evo_pointcloud.getWidth() * evo_pointcloud.getHeight(); m++){
					PointT p;
					p.x = evo_pointcloud.data[m * 4];			//X
					p.y = evo_pointcloud.data[m * 4 + 1];		//Y
					p.z = evo_pointcloud.data[m * 4 + 2];		//Z
					int * bgr_ptr = (int*)evo_pointcloud.data + m * 4 + 3;	//B-G-R-A
					p.rgba = *bgr_ptr;
					cloud->push_back(p);	//add point to cloud
				}
				//show point cloud
				viewer->showCloud(cloud);
			}
		}

		cloud->points.clear();
	}

	return 0;
}