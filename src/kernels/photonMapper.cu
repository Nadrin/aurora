/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <kernels/kernels.h>

using namespace Aurora;

#include <kernels/lib/common.cuh>
#include <kernels/lib/sampling.cuh>
#include <kernels/lib/intersect.cuh>
#include <kernels/lib/bsdf.cuh>
#include <kernels/lib/shader.cuh>

__global__ static void cudaRaycastPrimaryKernel(const Geometry geometry,
	const unsigned int numRays, Ray* rays, HitPoint* hitpoints)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= numRays)
		return;

	rays[threadId].t = Infinity;
	if(intersect(geometry, rays[threadId], hitpoints[threadId]))
		hitpoints[threadId].position = rays[threadId].point();
	else
		hitpoints[threadId].triangleID = -1;
}

__host__ void cudaRaycastPrimary(const Geometry& geometry, const unsigned int numRays, Ray* rays, HitPoint* hitpoints)
{
	dim3 blockSize(256);
	dim3 gridSize = make_grid(blockSize, dim3(numRays));
	cudaRaycastPrimaryKernel<<<gridSize, blockSize>>>(geometry, numRays, rays, hitpoints);
}

__global__ static void cudaGeneratePhotons(RNG* grng, const Geometry geometry, const Shader* shaders,
	const unsigned int numLights, const PolyLight* lights,
	const unsigned int numPhotons, Photon* photons)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= numPhotons)
		return;

	RNG rng = grng[threadId];
	const unsigned int lightIndex = sampleLightArray(curand_uniform(&rng), numLights, lights);
	const unsigned int triangleID = lights[lightIndex].triangleID;

	float u, v;
	sampleTriangle(curand_uniform(&rng), curand_uniform(&rng), u, v);

	const float3 P = getPosition(geometry, triangleID, u, v);
	const float3 N = getNormal(geometry, triangleID, u, v);

	const Shader& shader = shaders[getSafeID(geometry.shaders[triangleID])];

	photons[threadId].pos   = P;
	photons[threadId].dir   = N;
	photons[threadId].power = shader.color;
}

__global__ void cudaDebugPhotons(const unsigned int numHitPoints, HitPoint* hitpoints,
	const unsigned int numPhotons, const Photon* photons)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= numHitPoints)
		return;

	hitpoints[threadId].color = make_float3(0.5f, 0.5f, 0.5f);

	const float3 P = hitpoints[threadId].position;
	for(unsigned int i=0; i<numPhotons; i++) {
		if(distance(photons[i].pos, P) < 0.1f) {
			hitpoints[threadId].color = make_float3(1.0f, 1.0f, 1.0f);
			break;
		}
	}
}

__host__ void cudaPhotonTrace(RNG* rng, const Geometry& geometry, const ShadersArray& shaders, 
	const unsigned int numLights, const PolyLight* lights,
	const unsigned int numPhotons, Photon* photons,
	const unsigned int numHitPoints, HitPoint* hitpoints)
{
	dim3 blockSize, gridSize;

	if(numLights == 0)
		return;

	// Generate photons
	blockSize = dim3(256);
	gridSize  = make_grid(blockSize, dim3(numPhotons));

	cudaGeneratePhotons<<<gridSize, blockSize>>>(
		rng, geometry, shaders.items, 
		numLights, lights, numPhotons, photons);

	gridSize = make_grid(blockSize, dim3(numHitPoints));
	cudaDebugPhotons<<<gridSize, blockSize>>>(numHitPoints, hitpoints, numPhotons, photons);
}