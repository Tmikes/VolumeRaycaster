__global__ void updateCircleData(int pNbr, int pSegments, float* pCenters, float* pVertexData, float* pCenterData) {

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	int id = x + y * gridDim.x + z * gridDim.x*gridDim.y;

	if (id < pNbr)
	{
		int nbrCircles = pNbr / pSegments;
		int i = id % pSegments;
		float const t1 = 2 * M_PI * (float)i / (float)pSegments;
		float const t2 = 2 * M_PI * (float)(i + 1) / (float)pSegments;

		// vertices
		pVertex[id] = pCenters[];
		pVertex.push_back(pY);
		pVertex.push_back(pX + sin(t1) * pA);
		pVertex.push_back(pY + cos(t1) * pB);
		pVertex.push_back(pX + sin(t2) * pA);
		pVertex.push_back(pY + cos(t2) * pB);


		//center

		mCenter.push_back(pX);
		mCenter.push_back(pY);
		mCenter.push_back(pX);
		mCenter.push_back(pY);
		mCenter.push_back(pX);
		mCenter.push_back(pY);
	}

}