
__global__ void blur(float* source, float *output, int3 pDim, int r)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;
	int l = 2 * r + 1;
	if (x < pDim.x && y < pDim.y && z < pDim.z)
	{
		float kerneltotal = 0;
		int i = 0;
		if (source[x + pDim.x * y + pDim.x * pDim.y*z] != 0)
		{
			for (int xi = x - r, xii = 0; xi <= x + r; xi++, xii++)
			{
				for (int yi = y - r, yii = 0; yi <= y + r; yi++, yii++)
				{
					for (int zi = z - r, zii = 0; zi <= z + r; zi++, zii++)
					{
						if (xi >= 0 && xi < pDim.x && yi >= 0 && yi < pDim.y && zi >= 0 && zi < pDim.z)
						{
							//float source = 
							output[x + pDim.x * y + pDim.x * pDim.y*z] += (source[xi + pDim.x*yi + pDim.x*pDim.y*zi]) / ((2 * r + 1)*(2 * r + 1)*(2 * r + 1));
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

extern "C" void blurData(std::vector<float> pInput, std::vector<float>& pOutput, int3 pDim, int pRadius) {
	float *dev_input = 0;
	float *dev_output = 0;
	int size = pDim.x * pDim.y*pDim.z;
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

	dim3 blockSize(8, 8, 1);
	dim3 gridSize(iDivUp(pDim.x, blockSize.x), iDivUp(pDim.y, blockSize.y), iDivUp(pDim.z, blockSize.z));

	// Launch a kernel on the GPU with one thread for each element.
	blur << < gridSize, blockSize >> > (dev_input, dev_output, pDim, pRadius);

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