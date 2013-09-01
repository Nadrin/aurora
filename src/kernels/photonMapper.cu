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
#include <kernels/lib/light.cuh>
#include <kernels/lib/emitter.cuh>
#include <kernels/lib/radiance.cuh>

__global__ static void cudaRaycastPrimaryKernel(const Geometry geometry,
	const unsigned int numHitPoints, const Ray* rays, HitPoint* hitpoints)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= numHitPoints)
		return;

	Ray ray = rays[threadId];
	ray.t   = Infinity;
	if(intersect(geometry, ray, hitpoints[threadId])) {
		hitpoints[threadId].position = ray.point();
		hitpoints[threadId].wo       = -ray.dir;
	}
	else
		hitpoints[threadId].triangleID = -1;
}

__host__ void cudaRaycastPrimary(const PhotonMapperParams& params, const Geometry& geometry,
	Ray* rays, HitPoint* hitpoints)
{
	dim3 blockSize(256);
	dim3 gridSize = make_grid(blockSize, dim3(params.numHitPoints));
	cudaRaycastPrimaryKernel<<<gridSize, blockSize>>>(geometry, params.numHitPoints, rays, hitpoints);
}

__global__ static void cudaGeneratePhotonsFromLights(RNG* grng, const Geometry geometry,
	const unsigned int numLights, const Light* lights, const float* lightCDF,
	const unsigned int numPhotons, Photon* photons)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= numPhotons)
		return;

	RNG rng = grng[threadId];

	const unsigned int lightId = sampleDiscreteCDF(curand_uniform(&rng), numLights, lightCDF);
	photons[threadId] = lights[lightId].emitPhoton(&rng, geometry);
}

__global__ static void cudaGeneratePhotonsFromEmitters(RNG* grng, const Geometry geometry,
	const unsigned int numEmitters, const Emitter* emitters,
	const unsigned int numPhotons, Photon* photons)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= numPhotons)
		return;

	RNG rng = grng[threadId];
	const unsigned int emIndex    = sampleEmitters(curand_uniform(&rng), numEmitters, emitters);
	const unsigned int triangleID = emitters[emIndex].triangleID;

	float u, v;
	sampleTriangle(curand_uniform(&rng), curand_uniform(&rng), u, v);

	const float3 P = getPosition(geometry, triangleID, u, v);
	const float3 N = getNormal(geometry, triangleID, u, v);

	photons[threadId].pos    = P;
	photons[threadId].wi     = N;
	photons[threadId].energy = emitters[emIndex].power;
}

__global__ static void cudaTracePhotons(RNG* grng, const Geometry geometry, const Shader* shaders,
	const unsigned int numPhotons, const unsigned short maxDepth, Photon* photons)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= numPhotons)
		return;

	RNG rng       = grng[threadId];
	Photon photon = photons[threadId];

	HitPoint hp;
	unsigned short depth=0;

	while(true) {
		Ray ray(photon.pos, photon.wi);
		ray.offset();

		if(!intersect(geometry, ray, hp)) {
			photons[threadId].weight = 0.0f;
			return;
		}

		const Shader& shader = shaders[getSafeID(geometry.shaders[hp.triangleID])];
		const BSDF&   bsdf   = shader.getBSDF(geometry, hp.triangleID, hp.u, hp.v);

		float3 wo;
		float pdf;
		const float3 Kd = bsdf.samplef(&rng, photon.wi, wo, pdf);
		const float Pd  = luminosity(Kd * photon.energy) / luminosity(photon.energy);

		photon.energy = photon.energy * Kd;
		photon.pos    = ray.point();

		//if(depth++ > 0) {
			if(curand_uniform(&rng) > Pd || depth == maxDepth)
				break;
		//}
		photon.wi = wo;
	}

	photons[threadId] = photon;
}

__global__ static void cudaDebugPhotons(const unsigned int numHitPoints, HitPoint* hitpoints,
	const Geometry geometry, const unsigned int numPhotons, const Photon* photons)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= numHitPoints)
		return;

	HitPoint& hp = hitpoints[threadId];
	if(hp.triangleID == -1)
		return;

	const float3 P = hp.position;
	//const float3 N = getNormal(geometry, hp.triangleID, hp.u, hp.v);
	for(unsigned int i=0; i<numPhotons; i++) {
		const float d = distance(photons[i].pos, P);
		if(d < 0.1f) {
			const float nd = 1.0f - d / 0.1f;
			if(photons[i].weight > 0.0f)
				hp.color = hp.color + nd * photons[i].energy * photons[i].weight;
		}
	}
}

__global__ static void cudaRenderDirect(const PhotonMapperParams params, RNG* grng, 
	const Geometry geometry, const Shader* shaders, const Light* lights, HitPoint* hitpoints)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= params.numHitPoints)
		return;
	if(hitpoints[threadId].triangleID == -1)
		return;

	RNG rng      = grng[threadId];
	HitPoint& hp = hitpoints[threadId];

	const Shader& shader = shaders[getSafeID(geometry.shaders[hp.triangleID])];
	const BSDF&   bsdf   = shader.getBSDF(geometry, hp.triangleID, hp.u, hp.v);

	float3 Li = shader.emissionColor;
	for(unsigned int i=0; i<params.numLights; i++) {
		const Light& light = lights[i];
		if(light.isDeltaLight())
			Li = Li + estimateDirectRadianceDelta(geometry, light, shader, bsdf, hp);
		else
			Li = Li + estimateDirectRadiance(&rng, geometry, light, shader, bsdf, hp);
	}

	hp.color = Li;
}

__host__ void cudaPhotonTrace(const PhotonMapperParams& params, RNG* rng,
	const Geometry& geometry, const ShadersArray& shaders, const LightsArray& lights,
	const Emitter* emitters, Photon* photons, HitPoint* hitpoints)
{
	dim3 blockSize, gridSize;

	// Generate photons
	blockSize = dim3(256);
	gridSize  = make_grid(blockSize, dim3(params.numPhotons));

	/*if(params.numLights > 0) {
		cudaGeneratePhotonsFromLights<<<gridSize, blockSize>>>(
			rng, geometry, params.numLights, lights.items, params.lightCDF, params.numPhotons, photons);
	}*/

	//cudaTracePhotons<<<gridSize, blockSize>>>(rng, geometry, shaders.items,
	//	params.numPhotons, params.maxPhotonDepth, photons);

	blockSize = dim3(64);
	gridSize  = make_grid(blockSize, dim3(params.numHitPoints));
	cudaRenderDirect<<<gridSize, blockSize>>>(params, rng, geometry, shaders.items, lights.items, hitpoints);

	//gridSize = make_grid(blockSize, dim3(params.numHitPoints));
	//cudaDebugPhotons<<<gridSize, blockSize>>>(params.numHitPoints, hitpoints, geometry, params.numPhotons, photons);
}