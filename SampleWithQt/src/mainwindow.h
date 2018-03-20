
/// QT header
#include <QOpenGLWindow>
#include <QOpenGLFunctions_3_0>
#include <QtWidgets>

//Cuda header
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

//EvoBinoSDK header
#include "evo_depthcamera.h"//depth camera


class MainWindow : public QOpenGLWindow, protected QOpenGLFunctions_3_0
{
public:
	explicit MainWindow(UpdateBehavior updateBehavior = NoPartialUpdate, QOpenGLWindow *parent = 0);
	~MainWindow();

protected:
	void initializeGL() Q_DECL_OVERRIDE;
	void paintGL() Q_DECL_OVERRIDE;
	void keyPressEvent(QKeyEvent *e) Q_DECL_OVERRIDE;

private:
	evo::bino::DepthCamera camera;
	evo::bino::GrabParameters grab_parameters;
	bool running;
	evo::Mat<unsigned char> evo_image_gpu, evo_depth_gpu;
	int w, h;//image width/height
	//declare some ressources (GL texture ID, GL shader ID...)
	GLuint imageTex;
	GLuint depthTex;
	cudaGraphicsResource* pcuImageRes;
	cudaGraphicsResource* pcuDepthRes;
	void capture();
};