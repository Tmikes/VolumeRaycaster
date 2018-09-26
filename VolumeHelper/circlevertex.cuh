__global__ void updateCircleData_k(int pNbr, int pSegments, float* pCenterInput, float* pVertexData, float* pCenterData, float pA, float pB) {

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	int id = x + y * gridDim.x + z * gridDim.x*gridDim.y;

	if (id < pNbr)
	{
		int nbrCircles = pNbr / pSegments;
		int i = id % pSegments;
		int centerID = id / pSegments;
		float const t1 = 2 * M_PI * i / (float)pSegments;
		float const t2 = 2 * M_PI * (i + 1) / (float)pSegments;
		// vertices
		float cX = pCenterInput[2 * centerID];
		float cY = pCenterInput[2 * centerID + 1];
		pVertexData[id * 6] = cX; // X
		pVertexData[id * 6 + 1] = cY;  //Y
		pVertexData[id * 6 + 2] = cX + sin(t1) * pA;
		pVertexData[id * 6 + 3] = cY + cos(t1) * pB;
		pVertexData[id * 6 + 4] = cX + sin(t2) * pA;
		pVertexData[id * 6 + 5] = cY + cos(t2) * pB;

		//center
		pCenterData[id * 6] = cX;
		pCenterData[id * 6 + 1] = cY;
		pCenterData[id * 6 + 2] = cX;
		pCenterData[id * 6 + 3] = cY;
		pCenterData[id * 6 + 4] = cX;
		pCenterData[id * 6 + 5] = cY;

	}

}

extern "C" void updateCircleData(int pSegments, std::vector<float> pCenterInput, std::vector<float>& pVertexData, std::vector<float>& pCenterData, float pA, float pB) {
	float *dev_CenterInput = 0;
	float *dev_CenterData = 0;
	float *dev_VertexData = 0;
	//int size = pDim.x * pDim.y*pDim.z;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors  .
	cudaStatus = cudaMalloc((void**)&dev_CenterInput, pCenterInput.size() * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_CenterData, pCenterData.size() * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_VertexData, pVertexData.size() * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_CenterInput, pCenterInput.data(), pCenterInput.size() * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	int nbr = pCenterData.size() / 6;
	dim3 blockSize(8, 8, 1);
	dim3 gridSize(iDivUp(nbr, blockSize.x), iDivUp(1, blockSize.y), iDivUp(1, blockSize.z));

	// Launch a kernel on the GPU with one thread for each element.
	updateCircleData_k <<< gridSize, blockSize >>> (nbr, pSegments, dev_CenterInput, dev_VertexData, dev_CenterData, pA, pB);

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
	cudaStatus = cudaMemcpy(pVertexData.data(), dev_VertexData, pVertexData.size() * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(pCenterData.data(), dev_CenterData, pCenterData.size() * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_VertexData);
	cudaFree(dev_CenterData);

}