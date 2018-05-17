/**************************************************************************************************
** This sample simply shows how to configure and open the camera,                                **
** then print its serial number and SDK version, then close the camera.                          **
***************************************************************************************************/


//EvoBinoSDK header
#include "evo_stereocamera.h"

int main(int argc, char* argv[])
{
	// Create a camera object
	evo::bino::StereoCamera camera;

	// Open camera
	evo::RESULT_CODE res = camera.open(evo::bino::RESOLUTION_FPS_MODE_HD720_60);
	
	// If successed
	if (res == evo::RESULT_CODE_OK)
	{
		// Print serial number and SDK version
		std::cout << "serial number: " << camera.getSerialNumber() << std::endl;
		std::cout << "SDK version: " << camera.getSDKVersion() << std::endl;
		
		// Close camera
		camera.close();
	}
	else
	{
		std::cerr << "open camera failed: " << evo::result_code2str(res) << std::endl;
	}
	return 0;
}
