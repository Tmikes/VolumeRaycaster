
__global__ void logScaleData(uchar* pIinput, int withLog, int3 pDim) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;
	if (x < pDim.x && y < pDim.y && z < pDim.z) {
		float max = 255;
		uchar dens = pIinput[x + y * pDim.x + z * pDim.x*pDim.y];
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

extern "C" void logScale(std::vector<unsigned char> pInput, bool pWithLog) {
	unsigned char *dev_input = 0;
	int size = dim.x * dim.y*dim.z;
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
	dim3 gridSize(iDivUp(dim.x, blockSize.x), iDivUp(dim.y, blockSize.y), iDivUp(dim.z, blockSize.z));

	// Launch a kernel on the GPU with one thread for each element.
	logScaleData << < gridSize, blockSize >> > (dev_input, pWithLog ? 1 : 0, dim);

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