
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>

#include "float.h"
//#include "helper_cuda.h"
#include "volumehelper.h"
//#include "volumehelper.cuh"


#include <helper_cuda.h>
#include <helper_math.h>

typedef unsigned int  uint;
typedef unsigned char uchar;
//cudaArray *d_volumeArray = 0;
cudaArray *d_volumeArray;
cudaArray *d_transferFuncArray;
int size = 0;
int dimx = 0;
int dimy = 0;
int dimz = 0;

texture<uchar, cudaTextureType3D, cudaReadModeNormalizedFloat> volumeTex;
texture<float4, cudaTextureType3D, cudaReadModeElementType> transferTex;
surface<void, cudaSurfaceType3D> volumeSurf;
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);



typedef struct
{
	float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray
{
	float3 o;   // origin
	float3 d;   // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
	// compute intersection of ray with all six bbox planes
	float3 invR = make_float3(1.0f) / r.d;
	float3 tbot = invR * (boxmin - r.o);
	float3 ttop = invR * (boxmax - r.o);

	// re-order intersections to find smallest and largest on each aposWorld.xs
	float3 tmin = fminf(ttop, tbot);
	float3 tmax = fmaxf(ttop, tbot);

	// find the largest tmin and the smallest tmax
	float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
	float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
	float3 r;
	r.x = dot(v, make_float3(M.m[0]));
	r.y = dot(v, make_float3(M.m[1]));
	r.z = dot(v, make_float3(M.m[2]));
	return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
	float4 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = 1.0f;
	return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
	rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	rgba.w = __saturatef(rgba.w);
	return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

__global__ void logScaleData(uchar* input, int withLog, int dimx, int dimy, int dimz) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;
	if (x < dimx && y < dimy && z < dimz) {
		float max = 255;
		uchar dens = input[x + y * dimx + z * dimx*dimy];
		if (withLog)
		{
			float result = (float)dens;
			float maxLog = log(max + 1);
			result = round(255 * (log(result + 1) / maxLog));
			dens = (uchar)result;
		}
		surf3Dwrite(dens, volumeSurf, x * sizeof(uchar), y, z, cudaBoundaryModeClamp);
	}
}

__global__ void
d_render(uint *d_output, uint imageW, uint imageH, float density, float transferOffset, float3 dim)
{
	const int maxSteps = 500;
	const float tstep = 0.01f;
	const float opacityThreshold = 0.95f;
	const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
	const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	

	float u = (x / (float)imageW)*2.0f - 1.0f;
	float v = (y / (float)imageH)*2.0f - 1.0f;

	// calculate eye ray in world space
	Ray eyeRay;
	eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	eyeRay.d = normalize(make_float3(u, v, -2.0f));
	eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	if (!hit) {
		d_output[y*imageW + x] = rgbaFloatToInt(make_float4(0,0,0,0));
			return;
	}
	if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

										// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;
	float3 light = make_float3(1, 1, 1);
	float3 lightdir = mul(c_invViewMatrix, normalize(make_float3(1, 1, 0)) ) ;

	float attenuation = 1.0f;
	float shininess = 10.0f;
	float3 specularlight;
	float diffuseCoeff = 0.5f;
	float specularCoeff = 0.3f;
	float ambiantCoeff = 0.2f;
	float invdot , specPow = 10;

	for (int i = 0; i<maxSteps; i++)
	{
		// read from 3D texture
		// remap position to [0, 1] coordinates
		float3 posWorld = pos * 0.5f + 0.5f;
		float sample = tex3D(volumeTex, posWorld.x, posWorld.y, posWorld.z);
		//sample *= 64.0f;    // scale for 10-bit data

		// lookup in transfer function texture
		float4 col = tex3D(transferTex, sample, transferOffset,0);
		col.w *= density;

		float gradx = (tex3D(volumeTex, posWorld.x + 0.5f / dim.x, posWorld.y, posWorld.z) - tex3D(volumeTex, posWorld.x - 0.5f / dim.x, posWorld.y, posWorld.z)) / 2;
		float gradz = (tex3D(volumeTex, posWorld.x, posWorld.y, posWorld.z + 0.5f / dim.z) - tex3D(volumeTex, posWorld.x, posWorld.y, posWorld.z - 0.5f / dim.z)) / 2;
		float grady = (tex3D(volumeTex, posWorld.x, posWorld.y + 0.5f / dim.y, posWorld.z) - tex3D(volumeTex, posWorld.x, posWorld.y - 0.5f / dim.y, posWorld.z)) / 2;

		float4 n = make_float4(normalize(make_float3(gradx, grady, gradz)), 1.0f);
		//n = normalize(n);
		//n = mul4(c_invViewMatrix, n);
		float dotprod = lightdir.x * n.x + lightdir.y * n.y + lightdir.z * n.z; // lambertian

		dotprod = max(0.0f, dotprod);
		dotprod = min(1.0f, dotprod);

		float3 ambianlight = make_float3(col.x, col.y, col.z);
		float3 diffuselight = dotprod * make_float3(light.x *col.x, light.y *col.y, light.z*col.z);

		//float3 
		if (dotprod < 0)
		{
			specularlight = make_float3(0, 0, 0);
		}
		else {
			invdot = dot(reflect(-lightdir, make_float3(n.x, n.y, n.z)), eyeRay.d);
			//specularlight = attenuation* pow(max(0.0f, invdot), shininess)*make_float3(1, 1, 1);
		}

			diffuselight *= diffuseCoeff;
			specularlight = specularCoeff * pow(max(0.0f, invdot), specPow)*make_float3(1, 1, 1);
		

		//diffuselight *= col.w;
			ambianlight *= col.w; //* occlusion;
						  //specularlight *= col.w;
		float3 color = ambianlight + diffuselight + specularlight;
		color.x = fminf(color.x, 1);
		color.y = fminf(color.y, 1);
		color.z = fminf(color.z, 1);




		// "under" operator for back-to-front blending
		//sum = lerp(sum, col, col.w);

		// pre-multiply alpha
		col.x = color.x * col.w;
		col.y = color.y * col.w;
		col.z = color.z * col.w;
		// "over" operator for front-to-back blending
		sum = sum + col * (1.0f - sum.w);

		// eposWorld.xt early if opaque
		if (sum.w > opacityThreshold)
			break;

		t += tstep;

		if (t > tfar) break;

		pos += step;
	}
	// write output color
	d_output[y*imageW + x] = rgbaFloatToInt( sum );
}


__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}
__global__ void blur( float* source, float *output,  int dimx, int dimy, int dimz, int r)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;
	int l = 2 * r + 1;
	if (x < dimx && y < dimy && z < dimz)
	{
		float kerneltotal = 0;
		int i = 0;
		if (source[x + dimx * y + dimx * dimy*z] != 0)
		{
			for (int xi = x - r, xii = 0; xi <= x + r; xi++, xii++)
			{
				for (int yi = y - r, yii = 0; yi <= y + r; yi++, yii++)
				{
					for (int zi = z - r, zii = 0; zi <= z + r; zi++, zii++)
					{
						if (xi >= 0 && xi < dimx && yi >= 0 && yi < dimy && zi >= 0 && zi < dimz)
						{
							//float source = 
							output[x + dimx * y + dimx * dimy*z] += (source[xi + dimx*yi  +  dimx*dimy*zi])/((2*r+1)*(2*r+1)*(2*r+1));
							//output[x + dimx * y + dimx * dimy*z] += kernel[i] * source[xi + dimx * yi + dimx * dimy*zi];
							//kerneltotal += kernel[i];

						}
						i++;
					}
				}
			}
			if (kerneltotal != 0)
			{
				//  output[x + dimx*y + dimx*dimy*z] /= kerneltotal;
			}
		}
	}
}
int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

extern "C" void logScale( std::vector<unsigned char> pInput,  bool pWithLog ) {
	unsigned char *dev_input = 0;
	int size = dimx * dimy*dimz;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_input, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_input, pInput.data(), size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	dim3 blockSize(8, 8, 1);
	dim3 gridSize(iDivUp(dimx, blockSize.x), iDivUp(dimy, blockSize.y), iDivUp(dimz, blockSize.z));

	// Launch a kernel on the GPU with one thread for each element.
	logScaleData <<< gridSize, blockSize >>> (dev_input, pWithLog ?1:0, dimx ,dimy ,dimz);

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

Error:
	cudaFree(dev_input);

}

extern "C" void blurData(std::vector<float> pInput, std::vector<float>& pOutput, int pDimx, int  pDimy, int pDimz, int pRadius) {
	float *dev_input = 0;
	float *dev_output = 0;
	int size = dimx * dimy*dimz;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_input, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_output, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_input, pInput.data(), size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	dim3 blockSize(8,8,1);
	dim3 gridSize(iDivUp(dimx, blockSize.x), iDivUp(dimy, blockSize.y), iDivUp(dimz, blockSize.z));

	// Launch a kernel on the GPU with one thread for each element.
	blur <<< gridSize, blockSize >>> ( dev_input, dev_output, pDimx, pDimy, pDimz, pRadius);

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
	cudaStatus = cudaMemcpy(pOutput.data(), dev_output, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_output);
	cudaFree(dev_input);
	
}

extern "C" void initCuda( std::vector<unsigned char> h_volume, int width, int height, int depth)
{
	size = width *height*depth;
	dimx = width;
	dimy = height;
	dimz = depth;
	//create 3D global texture
	cudaChannelFormatDesc channelDescVolume = cudaCreateChannelDesc<unsigned char>();
	cudaExtent vol_dim = { dimx, dimy, dimz };
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
			{ 0.0, 0.0, 0.0, 0.0, },
			{ 1.0, 0.0, 0.0, 1.0, },
			{ 1.0, 0.5, 0.0, 1.0, },
			{ 1.0, 1.0, 0.0, 1.0, },
			{ 0.0, 1.0, 0.0, 1.0, },
			{ 0.0, 1.0, 1.0, 1.0, },
			{ 0.0, 0.0, 1.0, 1.0, },
			{ 0.0, 0.0, 1.0, 1.0, },
			{ 1.0, 0.0, 0.0, 1.0, },
		//----------		
			{ 0.0, 0.0, 0.0, 0.0, },
			{ 1.0, 0.0, 0.0, 0.0, },
			{ 1.0, 0.5, 0.0, 0.0, },
			{ 1.0, 1.0, 0.0, 1.0, },
			{ 0.0, 1.0, 0.0, 1.0, },
			{ 0.0, 1.0, 1.0, 1.0, },
			{ 0.0, 0.0, 1.0, 1.0, },
			{ 0.0, 0.0, 1.0, 1.0, },
			{ 1.0, 0.0, 0.0, 1.0, },
		//----------		
			{ 0.0, 0.0, 0.0, 0.0, },
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
//
//extern "C" void updateTF(unsigned char * colors, float index, dim3 blockSize, dim3 gridSize, float opacity, float* debug)
//{
//	float* dev_debug;
//	cudaMalloc((void**)&dev_debug, 5 * sizeof(float));
//	cudaMemcpy(dev_debug, debug, 5 * sizeof(float), cudaMemcpyHostToDevice);
//	updateColors<<<gridSize, blockSize>>>(colors, dev_volume, index, dimx, dimy, dimz, opacity,dev_debug);
//	// Check for any errors launching the kernel
//	cudaError_t cudaStatus = cudaGetLastError();
//	const char * msg = cudaGetErrorString(cudaStatus);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//	}
//	cudaMemcpy(debug, dev_debug, 5 * sizeof(float), cudaMemcpyDeviceToHost);
//	//debug[0] = 10;
//}

//
//
//extern"C" void cleanupcuda() {
//	cudaFree(dev_volume);
//	cudaFreeArray(d_transferFuncArray);
//}

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


extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, unsigned int *d_output, unsigned int imageW, unsigned int imageH,
	float density, float transferOffset, float3 dim)
{
	d_render <<<gridSize, blockSize >>>(d_output, imageW, imageH, density,  transferOffset, dim);
}

extern "C" void copyInvViewMatrix(std::vector<float> pInvViewMatrix)
{
	checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, pInvViewMatrix.data(), sizeof(float)*pInvViewMatrix.size()));
}




