#pragma once
#pragma comment(lib, "VolumeHelper.lib")
#include <vector_types.h>
#include <vector>

extern "C" void addWithCuda(std::vector<int>  &c, const std::vector<int> a, const std::vector<int> b, unsigned int size);
extern "C" void freeCudaBuffers();


extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, unsigned int *d_output, unsigned int imageW, unsigned int imageH,float density, float transferOffset);

extern "C" void copyInvViewMatrix(std::vector<float> pInvViewMatrix);

//extern "C" void addWithCuda(std::vector<int> c, const std::vector<int> a, const std::vector<int> b, unsigned int size);
//extern "C" void initCuda(unsigned char *h_volume, int width, int height, int depth);
//extern "C" void updateTF(unsigned char* colors, float index,  dim3 blockSize, dim3 gridSize, float opacity,float* debug);
//extern"C" void cleanupcuda();