/**************************************************************************************************
** This sample demonstrates how to do obstacle detection using CUDA with the EvoBinoSDK          **
** Result view will be displayed with OpenCV GUI                                                 **
***************************************************************************************************/

/******   ======================
*****   * Keyboard shortcuts *
*****   ======================
*****    Key pressing is available when GUI window is in the front.
*****	 Press 1~5 to change result view mode.
*****    ________________________________________________________________________________________
*****   |                                               ||                                       |
*****   |               Main Hotkeys                    ||            Display Hotkeys            |
*****   |=======|=======================================||=======|===============================|
*****   | 'esc' | Exit this program                     || '1'   | Image With Obstacle           |
*****   | 'q'   | Show SDK version                      || '2'   | Image With Ground             |
*****   | 'w'   | Show stereo parameters                || '3'   | Top View                      |
*****   |'t/y'  | Change stixels parameters             || '4'   | Ground Map                    |
*****   |'u/i'  | Change top view parameters            || '5'   | Image With Stixels            |
*****   |'o/p'  | Change obstacle detection parameters  || '6'   | Stixels                       |
*****   |_______|_______________________________________||_______|_______________________________|*/

//standard header
#include <iostream>
#include <sstream>
#include <string>

//OpenCV header
#include "opencv2/opencv.hpp"

//EvoBinoSDK header
#include "evo_depthcamera.h"//depth camera
#include "evo_matconverter.h"//converter between evo::Mat and cv::Mat

evo::bino::DepthCamera camera;
int obstacleId = 1;//index for select obstacle detection result
evo::bino::StereoParameters stereoPara;//stereo parameter
bool running = true;//running flag
evo::bino::ObstacleDetectionParameters obstacle_parameters;//parameters for obstacle detection
evo::bino::ObstacleDetectionTopViewParameters obstacle_topview_parameters;//parameters for top view (if you are not retrieving this view, do not have to set this)
evo::bino::ObstacleDetectionStixelsParameters obstacle_stixels_parameters;//parameters for stixels (if you are not retrieving this view, do not have to set this)


//function for key press event
void handleKey(char key)
{
	switch (key)
	{
	case '1':
	case '2':
	case '3':
	case '4':
	case '5':
	case '6':
		obstacleId = key - '1';
		break;
	case 't':
		obstacle_stixels_parameters.colorStep = 10;
		obstacle_stixels_parameters.margin = evo::Size2_<int>(1, 1);
		obstacle_stixels_parameters.unitWidth = 10;
		obstacle_stixels_parameters.unitMinHeight = 10;
		camera.setObstacleDetectionStixelsParameters(obstacle_stixels_parameters);
		std::cout << "set stixels parameters 1" << std::endl;
		break;
	case 'y':
		obstacle_stixels_parameters.colorStep = 5;
		obstacle_stixels_parameters.margin = evo::Size2_<int>(0, 0);
		obstacle_stixels_parameters.unitWidth = 20;
		obstacle_stixels_parameters.unitMinHeight = 20;
		camera.setObstacleDetectionStixelsParameters(obstacle_stixels_parameters);
		std::cout << "set stixels parameters 2" << std::endl;
		break;
	case 'u':
		camera.setMeasureUnit(evo::bino::MEASURE_UNIT_METER);//change unit
		obstacle_parameters.obstacleMinSize = evo::Size2_<float>(0.02f, 0.05f);//min size of obstacle object
		obstacle_topview_parameters.sideRange = 1;//own size
		obstacle_topview_parameters.horizontalRange = 2.0f;//the range of x-axis of top view
		obstacle_topview_parameters.verticalRange = evo::Size2_<float>(0.0f, 5.0f);//the range of y-axis of top view
		obstacle_topview_parameters.scale = 100.0f;//real world distance * scale -> pixel
		obstacle_topview_parameters.colorRange = evo::Size3_<float>(0.5f, 1.5f, 3.0f);//color range
		camera.setObstacleDetectionParameters(obstacle_parameters);
		camera.setObstacleDetectionTopViewParameters(obstacle_topview_parameters);
		std::cout << "change unit to m, set top view parameters 1" << std::endl;
		break;
	case 'i':
		camera.setMeasureUnit(evo::bino::MEASURE_UNIT_MILLIMETER);//change unit
		obstacle_parameters.obstacleMinSize = evo::Size2_<float>(20, 50);//min size of obstacle object
		obstacle_topview_parameters.sideRange = 500;//own size
		obstacle_topview_parameters.horizontalRange = 3000;//the range of x-axis of top view
		obstacle_topview_parameters.verticalRange = evo::Size2_<float>(0, 4000);//the range of y-axis of top view
		obstacle_topview_parameters.scale = 0.1f;//real world distance * scale -> pixel
		obstacle_topview_parameters.colorRange = evo::Size3_<float>(400, 1000, 2000);//color range
		camera.setObstacleDetectionParameters(obstacle_parameters);
		camera.setObstacleDetectionTopViewParameters(obstacle_topview_parameters);
		std::cout << "change unit to mm, set top view parameters 2" << std::endl;
		break;
	case 'o':
		obstacle_parameters.threshold1 = 30;
		obstacle_parameters.threshold2 = 40;
		obstacle_parameters.delta = 0.005f;
		obstacle_parameters.beta = 0.003f;
		obstacle_parameters.obstacleMaxHeight = INFINITY;
		camera.setObstacleDetectionParameters(obstacle_parameters);
		std::cout << "set obstacle detection parameters 1" << std::endl;
		break;
	case 'p':
		obstacle_parameters.threshold1 = 40;
		obstacle_parameters.threshold2 = 60;
		obstacle_parameters.delta = 0.01f;
		obstacle_parameters.beta = 0.01f;
		obstacle_parameters.obstacleMaxHeight = 2000.0f;
		camera.setObstacleDetectionParameters(obstacle_parameters);
		std::cout << "set obstacle detection parameters 2" << std::endl;
		break;
	
	case 27://exit
		running = false;
		break;
	case 'q'://SDK version
		std::cout << "SDK version: " << camera.getSDKVersion() << std::endl;
		break;
	case 'w'://stereo parameter
		stereoPara = camera.getStereoParameters(true);
		std::cout << "baseline: " << stereoPara.baseline() << ", focal: " << stereoPara.leftCam.focal.x << std::endl;
		break;
	default:
		break;
	}
}

int main(int argc, char* argv[])
{
	std::cout << "usage: SampleObstacleDetection [using_IMU(0: false(default), 1: true)]" << std::endl << std::endl;

	//IMU flag
	bool using_imu = false;

	if (argc == 2)
	{
		if (argv[1][0] == '0')
		{
			using_imu = false;
		}
		else if (argv[1][0] == '1')
		{
			using_imu = true;
		}
	}
	if (using_imu)
	{
		std::cout << "***** Using IMU. There can be roll angle, but you should not use the camera in flip status." << std::endl;
		std::cout << "***** You may need to do IMU calibration first, or position result may be wrong." << std::endl;
	}
	else
	{
		std::cout << "***** Not using IMU. The camera should be almost parallel to the ground (without roll angle)." << std::endl;
	}
	std::cout << "***** There must be ground in the image, or obstacle detection can not be done." << std::endl << std::endl;

	//open camera
	evo::RESULT_CODE res = camera.open(evo::bino::RESOLUTION_FPS_MODE_HD720_60, 0, evo::bino::WORK_MODE_FAST);
	std::cout << "depth camera open: " << result_code2str(res) << std::endl;
	//show image size
	std::cout << "image width:" << camera.getImageSizeFPS().width << ", height:" << camera.getImageSizeFPS().height << std::endl;
	//grab parameters
	evo::bino::GrabParameters grab_parameters;
	grab_parameters.do_rectify = true;
	grab_parameters.calc_disparity = true;
	grab_parameters.calc_distance = true;
	grab_parameters.depth_mode = evo::bino::DEPTH_MODE_STANDARD;

	if (res == evo::RESULT_CODE_OK)//open camera successed
	{
		//obstacle detection result
		evo::bino::ObstacleDetectionResult od_result;
		//evo Mat
		evo::Mat<unsigned char> evo_result;//evo Mat for display
		//cv Mat
		cv::Mat cv_result, cv_result_bgr;
		//set callback
		cv::namedWindow("result");
		//start obstacle detection
		if (using_imu)
		{
			if (!camera.isIMUSupported())
			{
				std::cerr << "### This camera do not have IMU or reading IMU failed, start without IMU..." << std::endl;
				using_imu = false;
			}
			else
			{
				//start IMU retrieving before start obstacle detection
				res = camera.startRetrieveIMU();
				if (res == evo::RESULT_CODE_OK)//start IMU successed
				{
					using_imu = true;
				}
				else
				{
					std::cerr << "### startRetrieveIMU() failed, start without IMU..." << std::endl;
					using_imu = false;
				}
			}
		}
		//also you can use 9 axes here
		res = camera.startObstacleDetection(using_imu, false);
		
		if (res == evo::RESULT_CODE_OK)
		{
			//main loop
			while (running)
			{
				// Get frames and launch the computation
				if (camera.grab(grab_parameters) == evo::RESULT_CODE_OK)
				{
					//retrieve result
					od_result = camera.retrieveObstacleDetectionResult();
					//using the result to do something...
					//std::cout << od_result.toString() << std::endl;

					//retrieve view
					//retrieveObstacleDetectionView() returns a RGBA Mat, so when using OpenCV to show it, we need to swap R and B channel
					evo_result = camera.retrieveObstacleDetectionView((evo::bino::OBSTACLE_VIEW_TYPE)obstacleId, evo::MAT_TYPE_CPU);

					//Mat convert
					cv_result = evo::evoMat2cvMat(evo_result);
					if (cv_result.channels() == 4)
					{
						cv::cvtColor(cv_result, cv_result_bgr, CV_RGBA2BGR);//need to swap R and B channel for OpenCV display
					}
					else//ground map is 1 channel
					{
						cv_result_bgr = cv_result.clone();
					}

					//show result
					cv::imshow("result", cv_result_bgr);
				}
				//handle key press event
				handleKey((char)cv::waitKey(10));
			}
			//stop obstacle detection (you can skip this and call close() directly)
			camera.stopObstacleDetection();
		}
		else
		{
			std::cerr << "start obstacle detection failed: " << evo::result_code2str(res) << std::endl;
		}
		//close camera
		camera.close();
	}
	else
	{
		std::cerr << "open camera failed: " << evo::result_code2str(res) << std::endl;
	}
	return 0;
}
