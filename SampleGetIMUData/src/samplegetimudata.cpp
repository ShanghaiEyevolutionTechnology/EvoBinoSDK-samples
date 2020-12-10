﻿﻿/***************************************************************************************************************
** This sample simply shows how to get IMU data for 10 seconds.                                               **
** There are 4 evo::imu::IMU_DATA_TYPE:                                                                       **
**        evo::imu::IMU_DATA_TYPE_RAW, evo::imu::IMU_DATA_TYPE_RAW_CALIBRATED,                                **
**        evo::imu::IMU_DATA_TYPE_POSITION_6_AXES and evo::imu::IMU_DATA_TYPE_POSITION_9_AXES.                **
** When using evo::imu::IMU_DATA_TYPE_RAW,                                                                    **
**        only `raw_value`, `temperature`, `timestamp` will be filled.                                        **
** When using evo::imu::IMU_DATA_TYPE_RAW_CALIBRATED,                                                         **
**        `raw_value`, `raw_calibrated_value`, `temperature`, `timestamp` will be filled.                     **
** When using evo::imu::IMU_DATA_TYPE_POSITION_6_AXES,                                                        **
**        `raw_value`, `raw_calibrated_value`, `position_6_value`, `temperature`, `timestamp` will be filled. **
** When using evo::imu::IMU_DATA_TYPE_POSITION_9_AXES,                                                        **
**        `raw_value`, `raw_calibrated_value`, `position_9_value`, `temperature`, `timestamp` will be filled. **
** IMU may need calibration before getting position data.                                                     **
***************************************************************************************************************/

#include <iostream>
#include <thread>
#include <iomanip>
#define _USE_MATH_DEFINES 
#include <math.h>

//EvoBinoSDK header
// We use StereoCamera here because we only want to get IMU data, and do not need depth calculation.
#include "evo_stereocamera.h"

//Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

#define TIME_INTERVAL 10

int main(int argc, char* argv[])
{
	std::cout << "usage: SampleGetIMUData [data type(0: IMU_DATA_TYPE_RAW(default), 1: IMU_DATA_TYPE_RAW_CALIBRATED, 2: IMU_DATA_TYPE_POSITION_6_AXES 3: IMU_DATA_TYPE_POSITION_9_AXES]" << std::endl;
	std::cout << "you may need to do IMU calibration first, or position result may be wrong." << std::endl;
	
	// Define IMU data type
	evo::imu::IMU_DATA_TYPE data_type = evo::imu::IMU_DATA_TYPE_RAW;

	if (argc == 2)
	{
		if (argv[1][0] == '0')
		{
			data_type = evo::imu::IMU_DATA_TYPE_RAW;
		}
		else if (argv[1][0] == '1')
		{
			data_type = evo::imu::IMU_DATA_TYPE_RAW_CALIBRATED;
		}
		else if (argv[1][0] == '2')
		{
			data_type = evo::imu::IMU_DATA_TYPE_POSITION_6_AXES;
		}
		else if (argv[1][0] == '3')
		{
			data_type = evo::imu::IMU_DATA_TYPE_POSITION_9_AXES;
		}
		std::cout << "select IMU data type: " << evo::imu::imu_data_type2str(data_type) << std::endl;
	}

	// Create a camera object
	evo::bino::StereoCamera camera;

	// Open camera
	evo::RESULT_CODE res = camera.open(evo::bino::RESOLUTION_FPS_MODE_HD720_60);

	// Since we only want to get IMU data, set not do recify to speed up the process, depth calculation will not be done too
	evo::bino::GrabParameters grab_parameters;
	grab_parameters.do_rectify = false;
	
	// If successed
	if (res == evo::RESULT_CODE_OK)
	{
		// Print serial number and SDK version
		std::cout << "serial number: " << camera.getSerialNumber() << std::endl;
		std::cout << "SDK version: " << camera.getSDKVersion() << std::endl;

		// Set IMU data type
		camera.setIMUDataType(data_type);

		// Set IMU retrieve mode
		camera.setIMUDataRetrieveMode(evo::imu::IMU_DATA_RETRIEVE_MODE_NEWEST_IMAGE);
		
		// Start retrieve IMU data
		res = camera.startRetrieveIMU();

		// If successed
		if (res == evo::RESULT_CODE_OK)
		{
			// Time point
			std::chrono::time_point<std::chrono::high_resolution_clock> now, start;
			start = std::chrono::high_resolution_clock::now();

			while (true)
			{
				// Grab image (if you use evo::imu::IMU_DATA_RETRIEVE_MODE_NEWEST_IMAGE, you must grab image)
				res = camera.grab(grab_parameters);
				if (res == evo::RESULT_CODE_OK)
				{
					// Retrieve image
					evo::Mat<unsigned char> left = camera.retrieveImage(evo::bino::SIDE_LEFT);				

					// Retrieve IMU data
					std::vector<evo::imu::IMUData> vector_data = camera.retrieveIMUData();

					// Print newest result
					if (vector_data.size() > 0)
					{
						evo::imu::IMUData data = vector_data.at(vector_data.size() - 1);

						if (data_type == evo::imu::IMU_DATA_TYPE_RAW)
						{
							std::cout << "raw time/accel/gyro/magnet:\t" 
								<< data.timestamp << "\t"
								<< std::setprecision(4) << std::fixed
								<< data.accel[0] << " " << data.accel[1] << " " << data.accel[2] << "\t"
								<< data.gyro[0] << " " << data.gyro[1] << " " << data.gyro[2] << "\t"
								<< data.mag[0] << " " << data.mag[1] << " " << data.mag[2]
								
								<< std::endl;
						}
						else if (data_type == evo::imu::IMU_DATA_TYPE_RAW_CALIBRATED)
						{
							std::cout << "calibrated time/accel/gyro/magnet:\t" 
								<< data.timestamp << "\t"
								<< std::setprecision(4) << std::fixed 
								<< data.accel_calibrated[0] << " " << data.accel_calibrated[1] << " " << data.accel_calibrated[2] << "\t"
								<< data.gyro_calibrated[0] << " " << data.gyro_calibrated[1] << " " << data.gyro_calibrated[2] << "\t"
								<< data.mag_calibrated[0] << " " << data.mag_calibrated[1] << " " << data.mag_calibrated[2]
								<< std::endl;
						}
						else if (data_type == evo::imu::IMU_DATA_TYPE_POSITION_6_AXES)
						{
							// Convert Quaternion to Euler
							Eigen::Quaternionf q(data.quaternion_6[3], data.quaternion_6[0], data.quaternion_6[1], data.quaternion_6[2]);
							Eigen::Matrix3f rotationMatrix = q.matrix().cast<float>();
							auto euler = rotationMatrix.eulerAngles(1, 0, 2);
							std::cout << "6 axes time/roll/pitch/yaw:\t" 
								<< data.timestamp << "\t"
								<< std::setprecision(4) << std::fixed
								<< euler[2] * 180.0f / M_PI << " " << euler[1] * 180.0f / M_PI << " " << euler[0] * 180.0f / M_PI
								<< std::endl;
						}
						else if (data_type == evo::imu::IMU_DATA_TYPE_POSITION_9_AXES)
						{
							// Convert Quaternion to Euler
							Eigen::Quaternionf q(data.quaternion_9[3], data.quaternion_9[0], data.quaternion_9[1], data.quaternion_9[2]);
							Eigen::Matrix3f rotationMatrix = q.matrix().cast<float>();
							auto euler = rotationMatrix.eulerAngles(1, 0, 2);
							std::cout << "9 axes time/roll/pitch/yaw:\t" 
								<< data.timestamp << "\t"
								<< std::setprecision(4) << std::fixed
								<< euler[2] * 180.0f / M_PI << " " << euler[1] * 180.0f / M_PI << " " << euler[0] * 180.0f / M_PI
								<< std::endl;
						}
					}
				}
				std::this_thread::sleep_for(std::chrono::microseconds(10000));

				now = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double> elapsed = now - start;
				if (elapsed.count() > TIME_INTERVAL)
				{
					break;
				}
			}
		}

		// Stop retrieve IMU data
		camera.stopRetrieveIMU();

		std::cout << std::endl << "run over" << std::endl;

		// Close camera
		camera.close();
	}
	else
	{
		std::cerr << "open camera failed: " << evo::result_code2str(res) << std::endl;
	}
	return 0;
}
