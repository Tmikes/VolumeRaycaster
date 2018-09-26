
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>

#include "float.h"
//#include "helper_cuda.h"
#include "volumehelper.h"
//#include "volumehelper.cuh"

#include <math.h>
#include <helper_cuda.h>
#include <helper_math.h>

typedef unsigned int  uint;
typedef unsigned char uchar;
//cudaArray *d_volumeArray = 0;
cudaArray *d_volumeArray;
cudaArray *d_transferFuncArray;
int size = 0;
int3 dim = { 0,0,0 };


texture<uchar, cudaTextureType3D, cudaReadModeNormalizedFloat> volumeTex;
texture<float4, cudaTextureType3D, cudaReadModeElementType> transferTex;
surface<void, cudaSurfaceType3D> volumeSurf;
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}


#include "raycaster.cuh"
#include "logscale.cuh"
#include "blurdata.cuh"
#include "circlevertex.cuh"

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}






extern "C" void initCuda( std::vector<unsigned char> h_volume, int3 pDim)
{
	size = pDim.x *pDim.y*pDim.z;
	dim = pDim;
	
	//create 3D global texture
	cudaChannelFormatDesc channelDescVolume = cudaCreateChannelDesc<unsigned char>();
	cudaExtent vol_dim = { pDim.x, pDim.y, pDim.z };
	cudaMalloc3DArray(&d_volumeArray, &channelDescVolume, vol_dim, cudaArraySurfaceLoadStore);
	// copy data to 3D array
	cudaMemcpy3DParms copyParamsVol = { 0 };
	copyParamsVol.srcPtr = make_cudaPitchedPtr(h_volume.data(), vol_dim.width * sizeof(unsigned char), vol_dim.width, vol_dim.height);
	copyParamsVol.dstArray = d_volumeArray;
	copyParamsVol.extent = vol_dim;
	copyParamsVol.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParamsVol);

	// set texture parameters
	volumeTex.normalized = true;                      // access with normalized texture coordinates
	volumeTex.filterMode = cudaFilterModeLinear;      // linear interpolation
	volumeTex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	volumeTex.addressMode[1] = cudaAddressModeClamp;
	//volumeTex.addressMode[2] = cudaAddressModeClamp;

	// Bind the array to the texture
	cudaBindTextureToArray(volumeTex, d_volumeArray, channelDescVolume);
	checkCudaErrors(cudaBindSurfaceToArray(volumeSurf, d_volumeArray));
	//---------------------transfer tex---------------------------------------------------------

	// create transfer function texture
	float4 transferFunc[] =
	{
		//----------		
			{ 1.0, 0.0, 0.0, 0.0, },
			{ 1.0, 0.0, 0.0, 1.0, },
			{ 1.0, 0.5, 0.0, 1.0, },
			{ 1.0, 1.0, 0.0, 1.0, },
			{ 0.0, 1.0, 0.0, 1.0, },
			{ 0.0, 1.0, 1.0, 1.0, },
			{ 0.0, 0.0, 1.0, 1.0, },
			{ 0.0, 0.0, 1.0, 1.0, },
			{ 1.0, 0.0, 0.0, 1.0, },
		//----------		
			{ 1.0, 0.0, 0.0, 0.0, },
			{ 1.0, 0.0, 0.0, 0.0, },
			{ 1.0, 0.5, 0.0, 0.0, },
			{ 1.0, 1.0, 0.0, 1.0, },
			{ 0.0, 1.0, 0.0, 1.0, },
			{ 0.0, 1.0, 1.0, 1.0, },
			{ 0.0, 0.0, 1.0, 1.0, },
			{ 0.0, 0.0, 1.0, 1.0, },
			{ 1.0, 0.0, 0.0, 1.0, },
		//----------		
			{ 1.0, 0.0, 0.0, 0.0, },
			{ 1.0, 0.0, 0.0, 0.0, },
			{ 1.0, 0.5, 0.0, 0.0, },
			{ 1.0, 1.0, 0.0, 0.0, },
			{ 0.0, 1.0, 0.0, 0.0, },
			{ 0.0, 1.0, 1.0, 0.5, },
			{ 0.0, 0.0, 1.0, 1.0, },
			{ 0.0, 0.0, 1.0, 1.0, },
			{ 1.0, 0.0, 0.0, 1.0, },
	};


	//cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
	//checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(transferFunc) / sizeof(float4), 1));
	//checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice));

	//transferTex.filterMode = cudaFilterModeLinear;
	//transferTex.normalized = true;    // access with normalized texture coordinates
	//transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

	//													 // Bind the array to the texture
	//checkCudaErrors(cudaBindTextureToArray(transferTex, d_transferFuncArray, channelDesc2));

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	cudaExtent tf_dim = { 9, 3, 1 };
	cudaMalloc3DArray(&d_transferFuncArray, &channelDesc, tf_dim);
	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(transferFunc, tf_dim.width * sizeof(float4), tf_dim.width, tf_dim.height);
	copyParams.dstArray = d_transferFuncArray;
	copyParams.extent = tf_dim;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);
	
	// set texture parameters
	transferTex.normalized = true;                      // access with normalized texture coordinates
	transferTex.filterMode = cudaFilterModeLinear;      // linear interpolation
	transferTex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	transferTex.addressMode[1] = cudaAddressModeClamp;

											  //Bind the array to the texture
	cudaBindTextureToArray(transferTex, d_transferFuncArray, channelDesc);
}


// Helper function for using CUDA to add vectors in parallel.
extern "C" void addWithCuda(std::vector<int>  &c, const std::vector<int> a, const std::vector<int> b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a.data(), size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b.data(), size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel <<<1, size >>> (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c.data(), dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	//return cudaStatus;
}

extern "C"
void freeCudaBuffers()
{
	checkCudaErrors(cudaFreeArray(d_volumeArray));
	checkCudaErrors(cudaFreeArray(d_transferFuncArray));
}


extern "C" void copyInvViewMatrix(std::vector<float> pInvViewMatrix)
{
	checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, pInvViewMatrix.data(), sizeof(float)*pInvViewMatrix.size()));
}




