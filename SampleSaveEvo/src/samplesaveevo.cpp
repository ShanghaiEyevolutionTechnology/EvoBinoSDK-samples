/**************************************************************************************************
** This sample simply shows how to save one .evo file with 100 frames.                           **
** The frames recorded in .evo file will be side-by-side.                                        **
***************************************************************************************************/

#include <iostream>
#include <thread>

//EvoBinoSDK header
// We use StereoCamera here for better saving speed.
#include "evo_stereocamera.h"

// Target frame number, for example : 100
int target_frame_number = 100;

// Target .evo file name
std::string filename = "test.evo";

int main(int argc, char* argv[])
{
	// Create a camera object
	evo::bino::StereoCamera camera;

	//set not do recify to speed up the process, depth calculation will not be done too
	evo::bino::GrabParameters grab_parameters;
	grab_parameters.do_rectify = false;

	// Open camera
	evo::RESULT_CODE res = camera.open(evo::bino::RESOLUTION_FPS_MODE_HD720_60);
	
	// If successed
	if (res == evo::RESULT_CODE_OK)
	{
		// Print serial number and SDK version
		std::cout << "serial number: " << camera.getSerialNumber() << std::endl;
		std::cout << "SDK version: " << camera.getSDKVersion() << std::endl;
		
		// Init recording using default compress
		// If you set 2nd parameter to true, you may save the .imu file at the same time
		res = camera.initRecording(filename.c_str(), false);

		// If you select to save IMU data at the same time, remember to start IMU retrieving
		//camera.startRetrieveIMU();

		// If successed
		if (res == evo::RESULT_CODE_OK)
		{
			std::cout << "start recording: " << std::endl;
			int count = 0;

			// Loop until we record frames of target number, or error is occured
			while (count < target_frame_number)
			{
				// Get frames, do not do CPU rectify, it may be slow on some device
				// Check the grab result, if result is evo::RESULT_CODE_NOT_A_NEW_FRAME, it means new frame is not come, just wait and grab again
				if (camera.grab(grab_parameters) == evo::RESULT_CODE_OK)
				{
					// You can retrieve image for displaying here, or just skip retrieving
					//evo::Mat<unsigned char> sbs = camera.retrieveImage(evo::bino::SIDE_SBS);

					// Record the frame
					res = camera.record();

					// If successed
					if (res == evo::RESULT_CODE_OK)
					{
						// Update count
						count++;

						std::cout << ".";
						std::cout.flush();
					}
					else
					{
						std::cerr << "record failed: " << evo::result_code2str(res) << std::endl;
						break;
					}
				}
				std::this_thread::sleep_for(std::chrono::microseconds(1000));
			}
		}
		else
		{
			std::cerr << "init recording failed: " << evo::result_code2str(res) << std::endl;
		}

		// End recording
		res = camera.endRecording();

		std::cout << std::endl << "record over" << std::endl;

		// Close camera
		camera.close();
	}
	else
	{
		std::cerr << "open camera failed: " << evo::result_code2str(res) << std::endl;
	}
	return 0;
}
