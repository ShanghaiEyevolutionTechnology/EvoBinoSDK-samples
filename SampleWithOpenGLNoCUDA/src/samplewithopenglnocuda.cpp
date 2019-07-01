/***************************************************************************************************
** This sample demonstrates how to grab images and disparity map with the EvoBinoSDK without CUDA **
** For the image, this sample shows how to show a gray image					                  **
** For the depth, this sample shows how to show a RGBA image                                      **
****************************************************************************************************/

/******   ======================
*****   * Keyboard shortcuts *
*****   ======================
*****    Key pressing is available when GUI window is in the front.
*****    _____________________________________________
*****   |                                            |
*****   |               Main Hotkeys                 |
*****   |=======|====================================|
*****   | 'esc' | Exit this program                  |
*****   | 'r'   | Switch fill mode                   |
*****   | 'f'   | Switch full screen                 |
*****   | 'p'   | Switch show FPS                    |
*****   |_______|____________________________________|*/

/// Glut
#include "GL/freeglut_std.h"
#include "GL/freeglut_ext.h"

//EvoBinoSDK header
#include "evo_stereocamera.h"

//high resolution clock
#include <chrono>

#include <sstream>

evo::bino::StereoCamera camera;
evo::bino::GrabParameters grab_parameters;
bool running = false;
bool is_gray = false;
bool fullscreen = false;
evo::Mat<unsigned char> evo_image_cpu, evo_depth_cpu;
int w, h;//image width/height
//declare some ressources (GL texture ID, GL shader ID...)
GLuint imageTex, depthTex;
std::chrono::time_point<std::chrono::high_resolution_clock> now, last;
bool show_fps = false;


//key press event
void handleKeypress(unsigned char key, int x, int y)
{
	switch (key) 
	{
	case 27://exit
		running = false;
		break;
	case 'r':
		if (grab_parameters.depth_mode == evo::bino::DEPTH_MODE_STANDARD)
		{
			grab_parameters.depth_mode = evo::bino::DEPTH_MODE_FILL;
		}
		else
		{
			grab_parameters.depth_mode = evo::bino::DEPTH_MODE_STANDARD;
		}
		break;
	case 'f':
		if (fullscreen)
		{
			//Configure Window Postion
			glutInitWindowPosition(0, 0);
			//Configure Window Size
			glutReshapeWindow(w, h / 2);
			fullscreen = false;
		}
		else
		{
			glutFullScreen();
			fullscreen = true;
		}
		break;
	case 'p':
		show_fps = !show_fps;
		break;
	}
}

void draw()
{
	evo::RESULT_CODE res = camera.grab(grab_parameters);

	if (res == evo::RESULT_CODE_OK)
	{
		//Retrieve image and normalized depth
		evo_image_cpu = camera.retrieveImage(evo::bino::SIDE_LEFT);//gray 1 channel
		evo_depth_cpu = camera.retrieveNormalizedDepth(evo::bino::DEPTH_TYPE_DISTANCE_Z_COLOR, 255, 0);//RGBA 4 channels

		//OpenGL Part
		glDrawBuffer(GL_BACK); //write to both BACK_LEFT & BACK_RIGHT
		glLoadIdentity();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

		//Draw Image Texture in Left Part of Side by Side
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, imageTex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, w, h, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, evo_image_cpu.data);
		glBegin(GL_QUADS);
		glColor3f(1.0f, 1.0f, 1.0f);
		glTexCoord2f(0.0, 1.0);
		glVertex2f(-1.0, -1.0);
		glTexCoord2f(1.0, 1.0);
		glVertex2f(0.0, -1.0);
		glTexCoord2f(1.0, 0.0);
		glVertex2f(0.0, 1.0);
		glTexCoord2f(0.0, 0.0);
		glVertex2f(-1.0, 1.0);
		glEnd();
		glBindTexture(GL_TEXTURE_2D, 0);
		glDisable(GL_TEXTURE_2D);

		//Draw Depth Texture in Right Part of Side by Side
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, depthTex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, evo_depth_cpu.data);
		glBegin(GL_QUADS);
		glColor3f(1.0f, 1.0f, 1.0f);
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
		glDisable(GL_TEXTURE_2D);

		if (show_fps)
		{
			last = now;
			now = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed = now - last;
			float fps = 1.0f / elapsed.count();
			std::ostringstream os_fps;
			os_fps << fps << " fps";
			glColor3f(0, 0.8f, 0);
			glRasterPos2f(20.0f / w * 2 - 1, -60.0f / h * 2 + 1);
			glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24, reinterpret_cast<const unsigned char*>(os_fps.str().c_str()));
		}

		//swap.
		glutSwapBuffers();
	}


	glutPostRedisplay();

	if (!running) {
		glutLeaveMainLoop();
		glutDestroyWindow(1);
		camera.close();
		glBindTexture(GL_TEXTURE_2D, imageTex);
		glDeleteTextures(1, &imageTex);
		glBindTexture(GL_TEXTURE_2D, depthTex);
		glDeleteTextures(1, &depthTex);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
}



int main(int argc, char* argv[])
{
	//open camera
	evo::RESULT_CODE res = camera.open(evo::bino::RESOLUTION_FPS_MODE_HD720_60);
	std::cout << "camera open: " << result_code2str(res) << std::endl;

	if (res == evo::RESULT_CODE_OK)
	{
		//get Image Size
		w = camera.getImageSizeFPS().width;
		h = camera.getImageSizeFPS().height;
		std::cout << "image width:" << w << ", height:" << h << std::endl;

		//init glut
		glutInit(&argc, argv);

		//Setting up The Display
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
		//Configure Window Postion
		glutInitWindowPosition(0, 0);

		//Configure Window Size
		glutInitWindowSize(w, h / 2);

		//Create Window
		glutCreateWindow("Eyevolution OpenGL Sample (No CUDA)");

		glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

		glEnable(GL_TEXTURE_2D);

		//Create and Register OpenGL Texture for Image (Gray - 1 Channel)
		glGenTextures(1, &imageTex);
		glBindTexture(GL_TEXTURE_2D, imageTex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, w, h, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);

		//Create and Register a OpenGL texture for Depth (RGBA - 4 Channels)
		glGenTextures(1, &depthTex);
		glBindTexture(GL_TEXTURE_2D, depthTex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);

		glutKeyboardFunc(handleKeypress);

		//Set Draw Loop
		running = true;
		glutDisplayFunc(draw);
		glutMainLoop();
	}
}
