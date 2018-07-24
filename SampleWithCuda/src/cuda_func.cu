#include "cuda_func.cuh"

__global__ void create_checkerboard_kernel(unsigned char *pImage, unsigned int width, unsigned int height, int square_size, uchar3 color1, uchar3 color2)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y >= height || x >= width)
		return;
	
	// fill the image, alternate the colors
	if (x % square_size < (square_size / 2))
	{
		if (y % square_size < (square_size / 2))
		{
			pImage[(y * width + x) * 3] = color1.x;
			pImage[(y * width + x) * 3 + 1] = color1.y;
			pImage[(y * width + x) * 3 + 2] = color1.z;
		}
		else
		{
			pImage[(y * width + x) * 3] = color2.x;
			pImage[(y * width + x) * 3 + 1] = color2.y;
			pImage[(y * width + x) * 3 + 2] = color2.z;
		}
	}
	else
	{
		if (y % square_size < (square_size / 2))
		{
			pImage[(y * width + x) * 3] = color2.x;
			pImage[(y * width + x) * 3 + 1] = color2.y;
			pImage[(y * width + x) * 3 + 2] = color2.z;
		}
		else
		{
			pImage[(y * width + x) * 3] = color1.x;
			pImage[(y * width + x) * 3 + 1] = color1.y;
			pImage[(y * width + x) * 3 + 2] = color1.z;
		}
	}
}

//fill an image with a chekcer_board (BGR)
void create_checkerboard(evo::Mat<unsigned char> image, int square_size)
{
	dim3 block(32, 32);
	dim3 grid((image.getWidth() + block.x - 1) / block.x, (image.getHeight() + block.y - 1) / block.y);

	// define the two colors of the checkerboard
	uchar3 color1 = make_uchar3(255, 255, 255);//white
	uchar3 color2 = make_uchar3(0, 255, 0);//green
	
	// call the kernel
	create_checkerboard_kernel << <grid, block >> >((unsigned char*)image.data, image.getWidth(), image.getHeight(), square_size, color1, color2);
}



__global__ void replace_image_by_distance_kernel(const unsigned char *pImage, const float* pDepth, const unsigned char *pBackground, unsigned char *result, const float max_value, const unsigned int width, const unsigned int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y >= height || x >= width)
		return;

	// get the depth of the current pixel
	float z_distance = pDepth[y * width + x];
	int index = (y * width + x) * 3;
	if (isfinite(z_distance) && (z_distance > max_value))
	{
		result[index] = pBackground[index];
		result[index + 1] = pBackground[index + 1];
		result[index + 2] = pBackground[index + 2];
	}
	else
	{
		result[index] = pImage[index];
		result[index + 1] = pImage[index + 1];
		result[index + 2] = pImage[index + 2];
	}
}

__global__ void replace_image_by_distance_gray_kernel(const unsigned char *pImage, const float* pDepth, const unsigned char *pBackground, unsigned char *result, const float max_value, const unsigned int width, const unsigned int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y >= height || x >= width)
		return;

	// get the depth of the current pixel
	float z_distance = pDepth[y * width + x];

	if (isfinite(z_distance) && (z_distance > max_value))
	{	
		int index = (y * width + x) * 3;
		result[index] = pBackground[index];
		result[index + 1] = pBackground[index + 1];
		result[index + 2] = pBackground[index + 2];
	}
	else
	{
		int index = y * width + x;
		result[index * 3] = pImage[index];
		result[index * 3+ 1] = pImage[index];
		result[index * 3+ 2] = pImage[index];
	}
}

//replace the current image by background if the distance if above the threshold
void replace_image_by_distance(evo::Mat<unsigned char> image, evo::Mat<float> distance_z, evo::Mat<unsigned char> background, evo::Mat<unsigned char> result, float max_value)
{
	dim3 block(32, 32);
	dim3 grid((image.getWidth() + block.x - 1) / block.x, (image.getHeight() + block.y - 1) / block.y);
	
	// call the kernel
	if (image.getChannels() == 1)
	{
		replace_image_by_distance_gray_kernel << <grid, block >> >((unsigned char*)image.data, (float*)distance_z.data, (unsigned char*)background.data, (unsigned char*)result.data, max_value, image.getWidth(), image.getHeight());
	}
	else if (image.getChannels() == 3)
	{
		replace_image_by_distance_kernel << <grid, block >> >((unsigned char*)image.data, (float *)distance_z.data, (unsigned char*)background.data, (unsigned char*)result.data, max_value, image.getWidth(), image.getHeight());
	}
}