/**************************************************************************************************
** This sample demonstrates how to use the depth to mask the current image with a background     **
** Image is displayed with OpenCV GUI                                                            **
***************************************************************************************************/

/******   ======================
*****   * Keyboard shortcuts *
*****   ======================
*****    Key pressing is available when GUI window is in the front.
*****    _____________________________________________
*****   |                                            |
*****   |               Main Hotkeys                 |
*****   |=======|====================================|
*****   | 'esc' | Exit this program                  |
*****   | '1'   | Increase max_z_distance            |
*****   | '2'   | Decrease max_z_distance            |
*****   |_______|____________________________________|*/

//standard header
#include <iostream>
#include <sstream>
#include <string>

//Cuda header
#include <cuda.h>
#include <cuda_runtime.h>

//opencv header
#include <opencv2/opencv.hpp>

//EvoBinoSDK header
#include "evo_depthcamera.h"//depth camera
#include "evo_matconverter.h"//converter between evo Mat and cv Mat

//Cuda functions
#include "cuda_func.cuh"

//flag
bool running;
//define a z-distance threshold, when z-distance is farther than it, the image will be replace by checkerboard
float max_z_distance = 4000.0f; //mm

#define MIN_DISTANCE 300.0f

//function for key press event
void handleKey(char key)
{
	int value = -1;
	switch (key)
	{
	case 27://exit
		running = false;
		break;
	case '1':
		max_z_distance += 100.0f;
		std::cout << "New distance threshold " << max_z_distance << " mm" << std::endl;
		break;
	case '2':
		max_z_distance -= 100.0f;
		if (max_z_distance < MIN_DISTANCE) max_z_distance = MIN_DISTANCE;
		std::cout << "New distance threshold " << max_z_distance << " mm" << std::endl; 
		break;
	default:
		break;
	}
}

int main(int argc, char* argv[])
{
	//variable
	evo::bino::DepthCamera camera;
	void *pCheckerboard_gpu, *pResult_gpu, *pResult_cpu;//pointer for checkerboard/result
	evo::Mat<unsigned char> evo_left_gpu, evo_checkerboard_gpu, evo_result_gpu, evo_result_cpu;
	evo::Mat<float> evo_z_gpu;
	unsigned int width, height;
	cv::Mat cv_result, cv_result_new;
	//open camera
	evo::RESULT_CODE res = camera.open(evo::bino::RESOLUTION_FPS_MODE_HD720_60);
	std::cout << "depth camera open: " << result_code2str(res) << std::endl;
	//grab parameter
	evo::bino::GrabParameters grab_parameters;
	//show image size
	width = camera.getImageSizeFPS().width;
	height = camera.getImageSizeFPS().height;
	std::cout << "image width:" << width << ", height:" << height << std::endl;
	//allocate
	cudaMalloc(&pCheckerboard_gpu, width * height * 3 * sizeof(unsigned char));
	cudaMalloc(&pResult_gpu, width * height * 3 * sizeof(unsigned char));
	pResult_cpu = malloc(width * height * 3 * sizeof(unsigned char));
	//link pointer and evo::Mat
	evo_checkerboard_gpu.setData(width, height, 3, (unsigned char*)pCheckerboard_gpu, evo::MAT_TYPE_GPU);
	evo_result_gpu.setData(width, height, 3, (unsigned char*)pResult_gpu, evo::MAT_TYPE_GPU);
	evo_result_cpu.setData(width, height, 3, (unsigned char*)pResult_cpu, evo::MAT_TYPE_CPU);
	//create checkerboard
	create_checkerboard(evo_checkerboard_gpu);

	std::cout << "Press Esc to quit, '1' to increase max_z_distance, '2' to decrease max_z_distance" << std::endl;

	if (res == evo::RESULT_CODE_OK)//open camera successed
	{
		//running flag
		running = true;

		//main loop
		while (running)
		{
			// Get frames and launch the computation
			if (camera.grab(grab_parameters) == evo::RESULT_CODE_OK)
			{
				//retrieve image
				evo_left_gpu = camera.retrieveImage(evo::bino::SIDE_LEFT, evo::MAT_TYPE_GPU);
				
				//retrieve distance
				evo_z_gpu = camera.retrieveDepth(evo::bino::DEPTH_TYPE_DISTANCE_Z, evo::MAT_TYPE_GPU);
				
				//do replace
				replace_image_by_distance(evo_left_gpu, evo_z_gpu, evo_checkerboard_gpu, evo_result_gpu, max_z_distance);

				//download
				cudaMemcpy(pResult_cpu, pResult_gpu, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

				//Mat convert
				cv_result = evo::evoMat2cvMat(evo_result_cpu);

				//show result
				cv::imshow("Result", cv_result);
			}
			//handle key press event
			handleKey((char)cv::waitKey(10));
		}
		//close camera
		camera.close();
		//deallocate
		cudaFree(pCheckerboard_gpu);
		cudaFree(pResult_gpu);
		free(pResult_cpu);
	}
	return 0;
}
