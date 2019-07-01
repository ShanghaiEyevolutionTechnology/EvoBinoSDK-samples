/**************************************************************************************************
** This sample simply shows how to save image sequences from .evo file.                          **
** Image sequences file named with frame number will be saved.                                   **
***************************************************************************************************/

//EvoBinoSDK header
#include "evo_stereocamera.h"
#include "evo_matconverter.h"//converter between evo mat and OpenCV mat

//OpenCV header
#include <opencv2/opencv.hpp>

//the bool value is used to work with the hotkey for breaking out
bool isRunning = true;

//press "ESC" or "q" to quit
void handleKey(char key)
{
	switch (key)
	{
	case 27:
	case 'q':
		isRunning = false;
		break;
	}
}


int main(int argc, char* argv[])
{
	// Create a camera object
	evo::bino::StereoCamera camera;

	//the user input: name of this program, file name of the evo file, do rectify or not(0 = no rectify, 1 = do rectify), start index of frame(int), end index of frame(int)
	std::cout << "usage: SampleEvoToImageSequences [. evo file, (only file name, without .evo)] [do rectify or not(default = 0, when value = 1, do rectify)][start index, default = 0][end index, default: end of the file]" << std::endl;

	//file name of the input evo file
	std::string file_name;
	std::string full_file_name;

	//the evo mat used to read evo file
	evo::Mat<unsigned char> image;

	//the OpenCV mat used to save image sequences file
	cv::Mat cvImage;

	//save the rectified image or raw image
	bool rectify = false;

	//the start frame index defined by user,default = 0
	int frame_index = 0;
	//the end frame index defined by user, default = the total frame number - 1
	//the index in the evo file is start from 0, so the last index in the file is "the total frame number - 1"
	int frame_index_last = 0;

	//check input information
	if (argc > 1)
	{
		file_name = argv[1];
		if (argc > 2)
		{
			if (atoi(argv[2]) == 1)
			{
				rectify = true;
			}
			if (argc > 3)
			{
				frame_index = atoi(argv[3]);
				if (argc > 4)
				{
					frame_index_last = atoi(argv[4]);
				}
			}
		}
	}
	else
	{
		return 0;
	}
	std::cout << "do rectify: " << rectify << std::endl;

	full_file_name = file_name + ".evo";

	std::cout << full_file_name << std::endl;

	//parameters for grabbing frame from .evo file
	evo::bino::GrabParameters grab_parameters;
	//this parameter will define do rectify or not
	grab_parameters.do_rectify = rectify;
	//since we only want to save image sequences, do not calculate disparity
	grab_parameters.calc_disparity = false;

	//open the evo file
	if (camera.open(full_file_name.c_str()) == evo::RESULT_CODE_OK)
	{
		//set the start index to camera
		camera.setTargetEvoPosition(frame_index);

		//choose the smaller one of input last frame number and the total frame number as the last frame number
		int lastIndex = camera.getEvoNumberOfFrames() - 1;
		if (argc > 4 && (frame_index_last < camera.getEvoNumberOfFrames() - 1))
		{
			lastIndex = frame_index_last;
		}

		//check if the process should be stopped
		while ((camera.getCurrentEvoPosition() < lastIndex) && isRunning)
		{
			//do grab before retrieve image everytime
			if (camera.grab(grab_parameters) == evo::RESULT_CODE_OK)
			{
				//retrieve image from evo file
				image = camera.retrieveImage(evo::bino::SIDE_SBS);

				//transform evo mat to OpenCV mat
				cvImage = evo::evoMat2cvMat(image);

				//get current frame index
				int index = camera.getCurrentEvoPosition();

				//generate image file name
				std::string image_file_name = file_name + "_" + std::to_string(index) + ".png";

				//write the frame to png file
				cv::imwrite(image_file_name, cvImage);

				//show current frame
				cv::imshow("image", cvImage);
			}
			//handle key event
			handleKey((char)cv::waitKey(10));
		}

		cv::waitKey(100);
		camera.close();
	}
	else
	{
		std::cout << "camera open failed" << std::endl;
	}

	return 0;
}
