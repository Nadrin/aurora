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

struct ValidEmitterPredicate
{
	__device__ bool operator()(const Emitter& e)
	{ return e.pdf > 0.0f; }
};

struct EmitterCdfBinaryOp
{
	__device__ Emitter operator()(const Emitter& a, const Emitter& b)
	{
		Emitter result;
		result.pdf        = b.pdf;
		result.power      = b.power;
		result.triangleID = b.triangleID;
		result.cdf        = a.cdf + b.cdf;
		return result;
	}
};

__global__ static void cudaInitEmitters(const Geometry geometry, const Shader* shaders, Emitter* emitters)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= geometry.count)
		return;

	const unsigned int shaderID = getSafeID(geometry.shaders[threadId]);

	emitters[threadId].power      = shaders[shaderID].emissionColor;
	emitters[threadId].pdf        = shaders[shaderID].emission;
	emitters[threadId].triangleID = threadId;
}

__global__ static void cudaCalculateEmittersCDF(const Geometry geometry,
	const unsigned int numEmitters, Emitter* lights)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= numEmitters)
		return;

	Primitive3 triangle;
	triangle.readPoints(geometry.vertices + lights[threadId].triangleID * Geometry::TriangleParams);
	
	const float area = triangle.area();
	lights[threadId].area = area;
	lights[threadId].cdf  = lights[threadId].pdf; // * area;
	lights[threadId].pdf  = 1.0f / area;
}

__global__ static void cudaNormalizeEmittersCDF(const unsigned int numEmitters, Emitter* lights,
	const float cdfIntegral)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= numEmitters)
		return;

	//lights[threadId].pdf /= (cdfIntegral * numEmitters);
	lights[threadId].cdf /= cdfIntegral;
}

unsigned int cudaCreateEmitters(const Geometry& geometry, const ShadersArray& shaders, Emitter** emitters)
{
	Emitter* buffer;

	*emitters = NULL;
	if(cudaMalloc(&buffer, sizeof(Emitter) * geometry.count) != cudaSuccess)
		return 0;
	if(cudaMalloc(emitters, sizeof(Emitter) * geometry.count) != cudaSuccess) {
		cudaFree(buffer);
		return 0;
	}

	dim3 blockSize = dim3(256);
	dim3 gridSize;

	gridSize  = make_grid(blockSize, dim3(geometry.count));
	cudaInitEmitters<<<gridSize, blockSize>>>(geometry, shaders.items, *emitters);

	thrust::device_ptr<Emitter> thrustBuffer(buffer);
	thrust::device_ptr<Emitter> thrustEmitters(*emitters);

	const auto thrustBufferEnd     = thrust::copy_if(thrustEmitters, thrustEmitters + geometry.count, thrustBuffer, ValidEmitterPredicate());
	const unsigned int numEmitters = thrustBufferEnd - thrustBuffer;
	if(numEmitters == 0) {
		cudaFree(buffer);
		return 0;
	}

	gridSize  = make_grid(blockSize, dim3(numEmitters));
	cudaCalculateEmittersCDF<<<gridSize, blockSize>>>(geometry, numEmitters, buffer);

	thrust::inclusive_scan(thrustBuffer, thrustBuffer + numEmitters, thrustEmitters, EmitterCdfBinaryOp());
	cudaFree(buffer);

	Emitter lastElement;
	cudaMemcpy(&lastElement, (*emitters) + (numEmitters-1), sizeof(Emitter), cudaMemcpyDeviceToHost);

	gridSize  = make_grid(blockSize, dim3(numEmitters));
	cudaNormalizeEmittersCDF<<<gridSize, blockSize>>>(numEmitters, *emitters, lastElement.cdf);
	return numEmitters;
}
