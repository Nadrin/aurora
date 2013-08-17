/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>

#include <kernels/kernels.h>
#include <kernels/intersect.h>

using namespace Aurora;

#include <kernels/common.cuh>
#include <kernels/intersect.cuh>

__global__ static void cudaRaytraceMonteCarloKernel(const Geometry geometry, const ShadersArray shaders, const LightsArray lights,
	const unsigned int numRays, Ray* rays, RNG* grng, float4* pixels)
{
	const unsigned int rayID = blockDim.x * blockIdx.x + threadIdx.x;
	if(rayID >= numRays)
		return;

	RNG    rng   = grng[threadIdx.x];
	float3 color = make_float3(0.0f, 0.0f, 0.0f);

	unsigned int triangleIndex;
	Ray ray = rays[rayID];

	if(!intersect(geometry, ray, triangleIndex)) {
		pixels[ray.id] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		return;
	}

	for(unsigned int i=0; i<lights.size; i++) {

	}

	pixels[ray.id] = make_float4(color.x, color.y, color.z, 1.0f);
}

__host__ void cudaRaytraceMonteCarlo(const Geometry& geometry, const ShadersArray& shaders, const LightsArray& lights,
	const unsigned int numRays, Ray* rays, RNG* rng, void* pixels)
{
	dim3 blockSize(256);
	dim3 gridSize = make_grid(blockSize, dim3(numRays));
	cudaRaytraceMonteCarloKernel<<<gridSize, blockSize>>>(geometry, shaders, lights, numRays, rays, rng, (float4*)pixels);
}