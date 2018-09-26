#pragma once
#pragma comment(lib, "VolumeHelper.lib")
#define _USE_MATH_DEFINES
#include <vector_types.h>
#include <vector>

extern "C" void addWithCuda(std::vector<int>  &c, const std::vector<int> a, const std::vector<int> b, unsigned int size);
extern "C" void freeCudaBuffers();


extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, unsigned int *d_output, unsigned int imageW, unsigned int imageH,float density, float transferOffset, float3 dim, float3 ratio);

extern "C" void copyInvViewMatrix(std::vector<float> pInvViewMatrix);

extern "C" void blurData(std::vector<float> pInput, std::vector<float>& pOutput, int3 pDim, int pRadius);

extern "C" void logScale(std::vector<unsigned char> pInput, bool pWithLog);
//extern "C" void addWithCuda(std::vector<int> c, const std::vector<int> a, const std::vector<int> b, unsigned int size);
extern "C" void initCuda(std::vector<unsigned char> h_volume, int3 pDim);

extern "C" void updateCircleData(int pSegments, std::vector<float> pCenterInput, std::vector<float>& pVertexData, std::vector<float>& pCenterData, float pA, float pB);


//extern "C" void updateTF(unsigned char* colors, float index,  dim3 blockSize, dim3 gridSize, float opacity,float* debug);
//extern"C" void cleanupcuda();