/**************************************************************************************************
** This sample simply shows how to transform one .evo file to .avi file.                         **
** Left view, right view and side-by-side avi file will be generated.                            **
***************************************************************************************************/

//header for the stereocamera
#include "evo_stereocamera.h"
//used to transform evo mat to OpenCV mat
#include "evo_matconverter.h"
//use OpenCV to save .avi file
#include <opencv2/core/core.hpp>

//the bool value is used to work with the hotkey for breaking out
bool isRunning = true;

//press "ESC" or "q" to quit
void handleKey(char key)
{
	int value = -1;
	switch (key)
	{
	case 27:
		isRunning = false;
		break;
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
	std::cout << "usage: SampleEvoToAvi [. evo file, (only file name, without .evo)] [do rectify or not(default = 0, when value = 1, do rectify)][start index, default = 0][end index, default: end of the file]" << std::endl;

	//file name of the input evo file
	std::string file_name;
	//file name of the sidy-by-side output avi file
	std::string file_name2;
	//file name of the left-view output avi file
	std::string file_name3;
	//file name of the right-view output avi file
	std::string file_name4;

	//the evo mat used to read evo file
	evo::Mat<unsigned char> image;
	evo::Mat<unsigned char> imageL;
	evo::Mat<unsigned char> imageR;

	//the cv mat used to save avi file
	cv::Mat cvImage;
	cv::Mat cvImageL;
	cv::Mat cvImageR;

	//the video write used to save avi file
	cv::VideoWriter writer;
	cv::VideoWriter writerL;
	cv::VideoWriter writerR;

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

	file_name2 = file_name;
	file_name3 = file_name;
	file_name4 = file_name;

	file_name.append(".evo");
	file_name2.append(".avi");
	file_name3.append("L.avi");
	file_name4.append("R.avi");

	std::cout << file_name << std::endl;

	//parameters for grabbing frame from .evo file
	evo::bino::GrabParameters grab_parameters;
	//this parameter will define do rectify or not
	grab_parameters.do_rectify = rectify;
	//since we only want to save .avi file, do not calculate disparity
	grab_parameters.calc_disparity = false;

	//open the evo file
	if (camera.open(file_name.c_str()) == evo::RESULT_CODE_OK)
	{
		//initial the avi file writer
		writer.open(file_name2, CV_FOURCC('M', 'P', '4', '2'), (double)camera.getImageSizeFPS().fps, cvSize(camera.getImageSizeFPS().width * 2, camera.getImageSizeFPS().height), false);
		writerL.open(file_name3, CV_FOURCC('M', 'P', '4', '2'), (double)camera.getImageSizeFPS().fps, cvSize(camera.getImageSizeFPS().width, camera.getImageSizeFPS().height), false);
		writerR.open(file_name4, CV_FOURCC('M', 'P', '4', '2'), (double)camera.getImageSizeFPS().fps, cvSize(camera.getImageSizeFPS().width, camera.getImageSizeFPS().height), false);

		if (!writer.isOpened() || !writerL.isOpened() || !writerR.isOpened())
		{
			std::cout << "create avi file failed" << std::endl;
			return 0;
		}

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
				imageL = camera.retrieveImage(evo::bino::SIDE_LEFT);
				imageR = camera.retrieveImage(evo::bino::SIDE_RIGHT);

				//transform evo mat to open cv mat
				cvImage = evo::evoMat2cvMat(image);
				cvImageL = evo::evoMat2cvMat(imageL);
				cvImageR = evo::evoMat2cvMat(imageR);

				//write the frame to avi file
				writer << cvImage;
				writerL << cvImageL;
				writerR << cvImageR;

				//show current frame
				cv::imshow("image", cvImage);

				handleKey((char)cv::waitKey(10));
			}
		}

		cv::waitKey(100);
		writer.release();
		writerL.release();
		writerR.release();
		camera.close();
	}
	else
	{
		std::cout << "camera open failed" << std::endl;
	}

	return 0;
}
