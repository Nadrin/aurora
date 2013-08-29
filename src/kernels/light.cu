/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <kernels/kernels.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

using namespace Aurora;

#include <kernels/lib/common.cuh>
#include <kernels/lib/sampling.cuh>
#include <kernels/lib/intersect.cuh>
#include <kernels/lib/light.cuh>


__global__ static void cudaComputeLightsPDF(const Geometry geometry, const unsigned int numLights, const Light* lights, float* cdf)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId < numLights)
		cdf[threadId] = lights[threadId].power(geometry);
}

__global__ static void cudaNormalizeLightsCDF(const unsigned int numLights, const float cdfIntegral, float* cdf)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId < numLights)
		cdf[threadId] /= cdfIntegral;
}

__host__ unsigned int cudaCreateLightsCDF(const Geometry& geometry, const LightsArray& lights, float** cdf)
{
	float* buffer;
	
	*cdf = NULL;
	if(cudaMalloc(&buffer, sizeof(float) * lights.size) != cudaSuccess)
		return 0;
	if(cudaMalloc(cdf, sizeof(float) * lights.size) != cudaSuccess) {
		cudaFree(buffer);
		return 0;
	}

	dim3 blockSize(256);
	dim3 gridSize = make_grid(blockSize, dim3(lights.size));

	cudaComputeLightsPDF<<<gridSize, blockSize>>>(geometry, lights.size, lights.items, buffer);

	thrust::device_ptr<float> thrustBuffer(buffer);
	thrust::device_ptr<float> thrustCdf(*cdf);
	thrust::inclusive_scan(thrustBuffer, thrustBuffer + lights.size, thrustCdf);
	cudaFree(buffer);

	float lastElement;
	cudaMemcpy(&lastElement, (*cdf) + (lights.size-1), sizeof(float), cudaMemcpyDeviceToHost);
	cudaNormalizeLightsCDF<<<gridSize, blockSize>>>(lights.size, lastElement, *cdf);
	return lights.size;
}