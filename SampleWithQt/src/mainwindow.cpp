/**************************************************************************************************
** This sample demonstrates how to use the EvoBinoSDK with Qt, OpenGL and CUDA at the same time  **
** The GPU buffer is ingested directly into OpenGL texture for avoiding GPU->CPU readback time   **
** For the image, this sample shows how to show a gray image                                     **
** For the depth, this sample shows how to show a RGBA image                                     **
***************************************************************************************************/

/******   ======================
*****   * Keyboard shortcuts *
*****   ======================
*****    Key pressing is available when QOpenGLWindow is in the front.
*****    _____________________________________________
*****   |                                            |
*****   |               Main Hotkeys                 |
*****   |=======|====================================|
*****   | 'esc' | Exit this program                  |
*****   |_______|____________________________________|*/

#include "mainwindow.h"

MainWindow::MainWindow(UpdateBehavior updateBehavior, QOpenGLWindow *parent) : QOpenGLWindow(updateBehavior, parent)
{
	//open camera
	evo::bino::RESOLUTION_FPS_MODE res_mode = evo::bino::RESOLUTION_FPS_MODE_HD720_60;
	evo::RESULT_CODE res = camera.open(res_mode);
	std::cout << "depth camera open:" << result_code2str(res) << std::endl;
	//get Image Size
	w = camera.getImageSizeFPS().width;
	h = camera.getImageSizeFPS().height;
	std::cout << "image width:" << w << ", height:" << h << std::endl;

	//running flag
	running = false;
	//init window parameter
	this->setTitle(tr("Eyevolution Qt Sample"));
	this->setHeight(h); 
	this->setWidth(w * 2);
}

MainWindow::~MainWindow()
{
	//close camera
	camera.close();
	
	glBindTexture(GL_TEXTURE_2D, 0);
	std::cout << "exit" << std::endl;
}

void MainWindow::initializeGL()
{
	initializeOpenGLFunctions();

	cudaError_t err1, err2;
	//Create and Register OpenGL Texture for Image (Gray - 1 Channel)
	glGenTextures(1, &imageTex);
	glBindTexture(GL_TEXTURE_2D, imageTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, w, h, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	err1 = cudaGraphicsGLRegisterImage(&pcuImageRes, imageTex, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
	
	//Create and Register a OpenGL texture for Depth (RGBA - 4 Channels)
	glGenTextures(1, &depthTex);
	glBindTexture(GL_TEXTURE_2D, depthTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	err2 = cudaGraphicsGLRegisterImage(&pcuDepthRes, depthTex, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);

	if (err1 != 0 || err2 != 0) std::cout << "CUDA resource register failed" << std::endl;

	glEnable(GL_TEXTURE_2D);

	running = true;
}

void MainWindow::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Escape)
	{
		running = false;
		close();
	}
}

void MainWindow::paintGL()
{
	if (running)
	{
		capture();
	}
}

void MainWindow::capture()
{
	evo::RESULT_CODE res = camera.grab(grab_parameters);

	if (res == evo::RESULT_CODE_OK)
	{
		//Map GPU Ressource for Image
		evo_image_gpu = camera.retrieveImage(evo::bino::SIDE_LEFT, evo::MAT_TYPE_GPU);//gray 1 channel

		cudaArray_t image_cuda_array;
		cudaGraphicsMapResources(1, &pcuImageRes, 0);
		cudaGraphicsSubResourceGetMappedArray(&image_cuda_array, pcuImageRes, 0, 0);
		cudaMemcpyToArray(image_cuda_array, 0, 0, evo_image_gpu.data, sizeof(unsigned char) * w * h, cudaMemcpyDeviceToDevice);
		cudaGraphicsUnmapResources(1, &pcuImageRes, 0);

		//Map GPU Ressource for Depth
		evo_depth_gpu = camera.retrieveNormalizedDepth(evo::bino::DEPTH_TYPE_DISTANCE_Z_COLOR, evo::MAT_TYPE_GPU, 255, 0);//RGBA 4 channels
		cudaArray_t depth_cuda_array;
		cudaGraphicsMapResources(1, &pcuDepthRes, 0);
		cudaGraphicsSubResourceGetMappedArray(&depth_cuda_array, pcuDepthRes, 0, 0);
		cudaMemcpyToArray(depth_cuda_array, 0, 0, evo_depth_gpu.data, sizeof(unsigned char) * w * h * 4, cudaMemcpyDeviceToDevice);
		cudaGraphicsUnmapResources(1, &pcuDepthRes, 0);

		//OpenGL Part
		glDrawBuffer(GL_BACK); //write to both BACK_LEFT & BACK_RIGHT
		glLoadIdentity();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		//Draw Image Texture in Left Part of Side by Side
		glBindTexture(GL_TEXTURE_2D, imageTex);

		glBegin(GL_QUADS);
		glTexCoord2f(0.0, 1.0);
		glVertex2f(-1.0, -1.0);
		glTexCoord2f(1.0, 1.0);
		glVertex2f(0.0, -1.0);
		glTexCoord2f(1.0, 0.0);
		glVertex2f(0.0, 1.0);
		glTexCoord2f(0.0, 0.0);
		glVertex2f(-1.0, 1.0);
		glEnd();

		glUseProgram(0);

		//Draw Depth Texture in Right Part of Side by Side
		glBindTexture(GL_TEXTURE_2D, depthTex);

		glBegin(GL_QUADS);
		glTexCoord2f(0.0, 1.0);
		glVertex2f(0.0, -1.0);
		glTexCoord2f(1.0, 1.0);
		glVertex2f(1.0, -1.0);
		glTexCoord2f(1.0, 0.0);
		glVertex2f(1.0, 1.0);
		glTexCoord2f(0.0, 0.0);
		glVertex2f(0.0, 1.0);
		glEnd();

		glBindTexture(GL_TEXTURE_2D, 0);
		glFinish();
	}
	update();
}
