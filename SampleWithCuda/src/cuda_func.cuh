#include <cuda_runtime.h>
#include "evo_mat.h"


//fill an image with a chekcer_board (BGR)
void create_checkerboard(evo::Mat<unsigned char> image, int square_size = 30);

//replace the current image by background if the distance if above the threshold
void replace_image_by_distance(evo::Mat<unsigned char> image, evo::Mat<float> distance_z, evo::Mat<unsigned char> background, evo::Mat<unsigned char> result, float max_value);
