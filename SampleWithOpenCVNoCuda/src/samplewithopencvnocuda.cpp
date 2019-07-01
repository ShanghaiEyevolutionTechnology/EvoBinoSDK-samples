/**************************************************************************************************
** This sample demonstrates how to grab images with the EvoBinoSDK without CUDA                  **
** Image is displayed with OpenCV GUI                                                            **
** Most of the functions of the SDK are linked with a key press event (using OpenCV)             **
***************************************************************************************************/

/******   ======================
*****   * Keyboard shortcuts *
*****   ======================
*****    Key pressing is available when GUI window is in the front.
*****	 Press 1~3 to change image mode.
*****    _____________________________________________________________________________________________
*****   |                                                    ||                                       |
*****   |               Main Hotkeys                         ||            Display Hotkeys            |
*****   |=======|============================================||=======|===============================|
*****   | 'esc' | Exit this program                          || '1'   | Left                          |
*****   | 'q'   | Show SDK version                           || '2'   | Right                         |
*****   | 'w'   | Show stereo parameters                     || '3'   | Side by Side                  |
*****   | 'e'   | Using the camera upside down               ||       |                               |
*****   | 'r'   | Save last image (image.png)                ||       |                               |
*****   | 't'   | Change doing rectify                       ||       |                               |
*****   | 'a'   | Change using auto exposure                 ||       |                               |
*****   | 's/d' | Change exposure time                       ||       |                               |
*****   |_______|____________________________________________||_______|_______________________________|*/

//standard header
#include <iostream>
#include <sstream>
#include <string>

//OpenCV header
#include <opencv2/opencv.hpp>

//EvoBinoSDK header
#include "evo_stereocamera.h"//stereo camera
#include "evo_matconverter.h"//converter between evo::Mat and cv::Mat

evo::bino::StereoCamera camera;
bool running = false, saveImage = false, autoExposure = true, doRectify = false;
int imageId = 1;//index for select image
evo::bino::StereoParameters stereoPara;//stereo parameter									   
evo::bino::GrabParameters grabPara;//grab parameter


//function for key press event
void handleKey(char key)
{
	float cur;
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
	case 27://exit
		running = false;
		break;
	case 'q'://SDK version
		std::cout << "SDK version: " << camera.getSDKVersion() << std::endl;
		break;
	case 'w'://stereo parameter
		stereoPara = camera.getStereoParameters(grabPara.do_rectify);
		std::cout << "rectify: " << grabPara.do_rectify << ", baseline: " << stereoPara.baseline() << ", focal: " << stereoPara.leftCam.focal.x << std::endl;
		break;
	case 'e'://flip
		camera.setFlip(!camera.getFlip());
		std::cout << "set camera flip inverse" << std::endl;
		break;
	case 'r'://save image
		saveImage = true;
		break;
	case 't'://do rectify
		grabPara.do_rectify = !grabPara.do_rectify;
		std::cout << "set do rectify: " << grabPara.do_rectify << std::endl;
		break;
	case 'a':
		autoExposure = !autoExposure;
		camera.useAutoExposure(autoExposure);
		std::cout << "set auto exposure: " << autoExposure << std::endl;
		break;
	case 's':
		cur = camera.getExposureTime();
		camera.setExposureTime(cur + 1);
		std::cout << "set exposure time: " << cur + 1 << std::endl;
		break;
	case 'd':
		cur = camera.getExposureTime();
		camera.setExposureTime(cur - 1);
		std::cout << "set exposure time: " << cur - 1 << std::endl;
		break;
	default:
		break;
	}
}

void print_help()
{
	std::cout << "======================" << std::endl;
	std::cout << "* Keyboard shortcuts *" << std::endl;
	std::cout << "======================" << std::endl;
	std::cout << " Key pressing is available when GUI window is in the front." << std::endl;
	std::cout << " Press 1~3 to change image mode." << std::endl;
	std::cout << " _____________________________________________________________________________________________" << std::endl;
	std::cout << "|                                                    ||                                       |" << std::endl;
	std::cout << "|               Main Hotkeys                         ||            Display Hotkeys            |" << std::endl;
	std::cout << "|=======|============================================||=======|===============================|" << std::endl;
	std::cout << "| 'esc' | Exit this program                          || '1'   | Left                          |" << std::endl;
	std::cout << "| 'q'   | Show SDK version                           || '2'   | Right                         |" << std::endl;
	std::cout << "| 'w'   | Show stereo parameters                     || '3'   | Side by Side                  |" << std::endl;
	std::cout << "| 'e'   | Using the camera upside down               ||       |                               |" << std::endl;
	std::cout << "| 'r'   | Save last image (image.png)                ||       |                               |" << std::endl;
	std::cout << "| 't'   | Change doing rectify                       ||       |                               |" << std::endl;
	std::cout << "| 'a'   | Change using auto exposure                 ||       |                               |" << std::endl;
	std::cout << "| 's/d' | Change exposure time                       ||       |                               |" << std::endl;
	std::cout << "|_______|____________________________________________||_______|_______________________________|" << std::endl;
}

int main(int argc, char* argv[])
{
	//open camera
	evo::bino::RESOLUTION_FPS_MODE res_mode = evo::bino::RESOLUTION_FPS_MODE_HD720_60;
	evo::RESULT_CODE res = camera.open(res_mode);
	std::cout << "stereo camera open: " << result_code2str(res) << std::endl;
	//show image size
	std::cout << "image width:" << camera.getImageSizeFPS().width << ", height:" << camera.getImageSizeFPS().height << std::endl;
	//grab parameters
	grabPara.do_rectify = false;
	grabPara.calc_disparity = false;

	if (res == evo::RESULT_CODE_OK)//open camera successed
	{
		print_help();
		//evo Mat
		evo::Mat<unsigned char> evo_image;//evo Mat for image display
		//cv Mat
		cv::Mat cv_image;
		//running flag
		running = true;
		//set callback
		cv::namedWindow("image");

		//main loop
		while (running)
		{
			// Get frames and launch the computation
			if (camera.grab(grabPara) == evo::RESULT_CODE_OK)
			{
				//retrieve image
				switch (imageId)
				{
				case 1://left
					evo_image = camera.retrieveImage(evo::bino::SIDE_LEFT);
					break;
				case 2://right
					evo_image = camera.retrieveImage(evo::bino::SIDE_RIGHT);
					break;
				case 3://side by side
					evo_image = camera.retrieveImage(evo::bino::SIDE_SBS);
					break;
				}
				
				//Mat convert
				cv_image = evo::evoMat2cvMat(evo_image);				
				
				//save
				if (saveImage)
				{
					cv::imwrite("image.png", cv_image);
					std::cout << "image saved" << std::endl;
					saveImage = false;
				}
								
				//show image and depth
				cv::imshow("image", cv_image);
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
