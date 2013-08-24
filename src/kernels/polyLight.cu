/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <kernels/kernels.h>

#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/scan.h>

using namespace Aurora;

#include <kernels/lib/common.cuh>
#include <kernels/lib/intersect.cuh>
#include <kernels/lib/bsdf.cuh>
#include <kernels/lib/shader.cuh>

struct ValidPolyLightPredicate
{
	__device__ bool operator()(const PolyLight& pl)
	{ return pl.emission > 0.0f; }
};

struct PolyLightCdfBinaryOp
{
	__device__ PolyLight operator()(const PolyLight& a, const PolyLight& b)
	{
		PolyLight result;
		result.emission   = b.emission;
		result.triangleID = b.triangleID;
		result.cdf        = a.emission + b.emission;
		return result;
	}
};

__global__ static void cudaInitPolyLights(const Geometry geometry, const ShadersArray shaders, PolyLight* lights)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= geometry.count)
		return;

	const unsigned int shaderID = getSafeID(geometry.shaders[threadId]);
	lights[threadId].triangleID = threadId;
	lights[threadId].emission   = shaders[shaderID].emission;
}

__global__ static void cudaCalculatePolyLightsCDF(const Geometry geometry, const unsigned int numLights, PolyLight* lights)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= numLights)
		return;

	volatile __shared__ float cdfIntegral;
	if(threadId == 0)
		cdfIntegral = lights[numLights-1].cdf;
	__syncthreads();
	lights[threadId].cdf /= cdfIntegral;

	Primitive3 triangle;
	triangle.readPoints(geometry.vertices + lights[threadId].triangleID * Geometry::TriangleParams);
	lights[threadId].area = triangle.area();
}

__host__ unsigned int cudaCreatePolyLights(const Geometry& geometry, const ShadersArray& shaders, PolyLight** lights)
{
	PolyLight* buffer;

	*lights = NULL;
	if(cudaMalloc(&buffer, sizeof(PolyLight) * geometry.count) != cudaSuccess)
		return 0;
	if(cudaMalloc(lights, sizeof(PolyLight) * geometry.count) != cudaSuccess) {
		cudaFree(buffer);
		return 0;
	}

	dim3 blockSize;
	dim3 gridSize;

	blockSize = dim3(512);
	gridSize  = make_grid(blockSize, geometry.count);
	cudaInitPolyLights<<<gridSize, blockSize>>>(geometry, shaders, *lights);

	thrust::device_ptr<PolyLight> thrustBuffer(buffer);
	thrust::device_ptr<PolyLight> thrustLights(*lights);

	const auto thrustBufferEnd   = thrust::copy_if(thrustLights, thrustLights + geometry.count, thrustBuffer, ValidPolyLightPredicate());
	const unsigned int numLights = thrustBufferEnd - thrustBuffer;
	if(numLights == 0) {
		cudaFree(buffer);
		return 0;
	}

	thrust::inclusive_scan(thrustBuffer, thrustBuffer + numLights, thrustLights, PolyLightCdfBinaryOp());
	cudaFree(buffer);

	blockSize = dim3(256);
	gridSize  = make_grid(blockSize, numLights);
	cudaCalculatePolyLightsCDF<<<gridSize, blockSize>>>(geometry, numLights, *lights);
	return numLights;
}
