/**************************************************************************************************
** This sample demonstrates how to grab images and depth map with the EvoBinoSDK                 **
** Both images and depth map are displayed with OpenCV GUI                                       **
** Most of the functions of the Bino SDK are linked with a key press event (using OpenCV)        **
***************************************************************************************************/

/******   ======================
*****   * Keyboard shortcuts *
*****   ======================
*****    Key pressing is available when GUI window is in the front.
*****	 Press 1~5 to change image view mode. Press 7~8 to change depth view mode.
*****    The distance of the point which mouse pressed is showed on the image.
*****    ________________________________________________________________________________________
*****   |                                               ||                                       |
*****   |               Main Hotkeys                    ||            Display Hotkeys            |
*****   |=======|=======================================||=======|===============================|
*****   | 'esc' | Exit this program                     || '1'   | Left                          |
*****   | 'q'   | Show SDK version                      || '2'   | Right                         |
*****   | 'w'   | Show stereo parameters                || '3'   | Side by Side                  |
*****   | 'e'   | Using the camera upside down          || '4'   | Anaglyph                      |
*****   | 'r'   | Save last image (image.png)           || '5'   | Overlay                       |
*****   | 't'   | Save last depth color (distance.png)  || '7'   | Distance                      |
*****   | 'y'   | Save last xyz (xyz.png)               || '8'   | Distance color                |
*****   |_______|_______________________________________||_______|_______________________________|*/

//standard header
#include <iostream>
#include <sstream>
#include <string>

//opencv header
#include "opencv2/opencv.hpp"

//EvoBinoSDK header
#include "evo_depthcamera.h"//depth camera
#include "evo_matconverter.h"//converter between evo::Mat and cv::Mat

evo::bino::DepthCamera camera;
bool running = false, saveImage = false, saveNormalizedDistanceZ = false, saveXYZ = false;
int imageId = 5, depthId = 2;//index for select image/depth
int mouseX, mouseY;//mouse coordinate
evo::bino::StereoParameters stereoPara;//stereo parameter


//function for mouse press event
static void onMouse(int event, int x, int y, int flags, void *param)
{
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		break;
	case CV_EVENT_LBUTTONUP:
		mouseX = x;
		mouseY = y;
		std::cout << "x: " << x << ", y: " << y << std::endl;
		break;
	case CV_EVENT_MOUSEMOVE:
		break;
	}
}

//function for key press event
void handleKey(char key)
{
	int value = -1;
	switch (key)
	{
	case '1'://left
		imageId = 1;
		break;
	case '2'://right
		imageId = 2;
		break;
	case '3'://side by side
		imageId = 3;
		break;
	case '4'://anaglyph
		imageId = 4;
		break;
	case '5'://overlay
		imageId = 5;
		break;
	case '7'://distance
		depthId = 1;
		break;
	case '8'://distance z color
		depthId = 2;
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
	case 'e'://flip
		camera.setFlip(!camera.getFlip());
		std::cout << "set camera flip inverse" << std::endl;
		break;
	case 'r'://save image
		saveImage = true;
		break;
	case 't'://save normalized distance_z color
		saveNormalizedDistanceZ = true;
		break;
	case 'y'://save syz
		saveXYZ = true;
		break;
	default:
		break;
	}
}

int main(int argc, char* argv[])
{
	//open camera
	evo::bino::RESOLUTION_FPS_MODE res_mode = evo::bino::RESOLUTION_FPS_MODE_HD720_60;
	evo::RESULT_CODE res = camera.open(res_mode);
	std::cout << "depth camera open: " << result_code2str(res) << std::endl;
	//show image size
	std::cout << "image width:" << camera.getImageSizeFPS().width << ", height:" << camera.getImageSizeFPS().height << std::endl;
	//set default point (center of the image) for showing distance
	mouseX = camera.getImageSizeFPS().width / 2;
	mouseY = camera.getImageSizeFPS().height / 2;
	//grab parameters
	evo::bino::GrabParameters grab_parameters;

	if (res == evo::RESULT_CODE_OK)//open camera successed
	{
		//evo Mat
		evo::Mat<unsigned char> evo_image, evo_depth;//evo Mat for image/depth display
		evo::Mat<float> evo_xyz;//evo Mat for depth
		//cv Mat
		cv::Mat cv_image, cv_image_bgr, cv_depth, cv_depth_bgr, cv_xyz;
		//running flag
		running = true;
		//set callback
		cv::namedWindow("image");
		cv::namedWindow("depth");
		cv::setMouseCallback("image", onMouse, NULL);
		cv::setMouseCallback("depth", onMouse, NULL);

		//main loop
		while (running)
		{
			// Get frames and launch the computation
			if (camera.grab(grab_parameters) == evo::RESULT_CODE_OK)
			{
				//retrieve image
				//retrieveView() returns a RGBA Mat, so when using OpenCV to show it, we need to swap R and B channel
				switch (imageId)
				{
				case 1://left
					evo_image = camera.retrieveView(evo::bino::VIEW_TYPE_LEFT, evo::MAT_TYPE_CPU);
					break;
				case 2://right
					evo_image = camera.retrieveView(evo::bino::VIEW_TYPE_RIGHT, evo::MAT_TYPE_CPU);
					break;
				case 3://side by side
					evo_image = camera.retrieveView(evo::bino::VIEW_TYPE_SBS, evo::MAT_TYPE_CPU);
					break;
				case 4://anaglyph
					evo_image = camera.retrieveView(evo::bino::VIEW_TYPE_ANAGLYPH, evo::MAT_TYPE_CPU);
					break;
				case 5://overlay
					evo_image = camera.retrieveView(evo::bino::VIEW_TYPE_OVERLAY, evo::MAT_TYPE_CPU);
					break;
				}

				//retrieve normalize depth for displaying
				//inverse the min/max value to have a clear view
				//if you want to get a disparity/distance map to do calculation, call retrieveDepth() to get a float type Mat
				switch (depthId)
				{
				case 1://distance z
					evo_depth = camera.retrieveNormalizedDepth(evo::bino::DEPTH_TYPE_DISTANCE_Z, evo::MAT_TYPE_CPU, 255, 0);
					break;
				case 2://distance z color
					//distance z color also has a float version (call retrieveDepth()), although it is useless...
					evo_depth = camera.retrieveNormalizedDepth(evo::bino::DEPTH_TYPE_DISTANCE_Z_COLOR, evo::MAT_TYPE_CPU, 255, 0);
					break;
				}
				
				//retrieve distance xyz
				evo_xyz = camera.retrieveDepth(evo::bino::DEPTH_TYPE_DISTANCE_XYZ, evo::MAT_TYPE_CPU);
				
				//Mat convert
				cv_image = evo::evoMat2cvMat(evo_image);
				cv::cvtColor(cv_image, cv_image_bgr, CV_RGBA2BGR);//need to swap R and B channel for OpenCV display

				cv_depth = evo::evoMat2cvMat(evo_depth);
				if (depthId == 2)//distance z color need to swap R and B channel for OpenCV display
				{
					cv::cvtColor(cv_depth, cv_depth_bgr, CV_RGBA2BGR);//rgba -> bgr
				}
				else
				{
					cv_depth_bgr = cv_depth.clone();//gray need no convert
				}

				cv_xyz = evo::evoMat2cvMat(evo_xyz);
				
				//save
				if (saveImage)
				{
					cv::imwrite("image.png", cv_image_bgr);
					std::cout << "image saved" << std::endl;
					saveImage = false;
				}
				if (saveNormalizedDistanceZ)
				{
					cv::imwrite("distance.png", cv_depth_bgr);
					std::cout << "normalized distance z saved" << std::endl;
					saveNormalizedDistanceZ = false;
				}
				if (saveXYZ)
				{
					cv::imwrite("xyz.png", cv_xyz);
					std::cout << "xyz saved" << std::endl;
					saveXYZ = false;
				}
				
				//write distance on image (x y z)
				std::ostringstream os_distance;
				os_distance << cv_xyz.at<cv::Vec3f>(mouseY, mouseX)[0] << " " << cv_xyz.at<cv::Vec3f>(mouseY, mouseX)[1] << " " << cv_xyz.at<cv::Vec3f>(mouseY, mouseX)[2];
				cv::circle(cv_image_bgr, cv::Point(mouseX, mouseY), 10, cv::Scalar(0, 200, 200), 3);
				cv::putText(cv_image_bgr, os_distance.str(), cv::Point(mouseX, mouseY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 200));
				cv::circle(cv_depth_bgr, cv::Point(mouseX, mouseY), 10, cv::Scalar(0, 200, 200), 3);
				cv::putText(cv_depth_bgr, os_distance.str(), cv::Point(mouseX, mouseY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 200));
				
				//show image and depth
				cv::imshow("image", cv_image_bgr);
				cv::imshow("depth", cv_depth_bgr);
			}
			//handle key press event
			handleKey((char)cv::waitKey(10));
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
