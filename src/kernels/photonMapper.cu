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

__global__ static void cudaGeneratePhotons(RNG* grng, const Geometry geometry,
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

	photons[threadId].pos   = P;
	photons[threadId].dir   = N;
	photons[threadId].power = emitters[emIndex].power;
}

__global__ static void cudaDebugPhotons(const unsigned int numHitPoints, HitPoint* hitpoints,
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

inline __device__ float3 estimateDirectRadianceDelta(const PhotonMapperParams& params,
	const Geometry& geometry, const Shader* shaders,
	const Light& light, const HitPoint& hp)
{
	const Shader& shader = shaders[getSafeID(geometry.shaders[hp.triangleID])];
	const BSDF&   bsdf   = shader.getBSDF(geometry, hp.triangleID, hp.u, hp.v);

	Ray wi(hp.position);

	float pdf;
	const float3 Le = shader.emission * shader.emissionColor;
	const float3 Li = light.sampleL(NULL, wi, pdf);
	wi.offset();

	if(pdf > 0.0f && !zero(Li)) {
		const float3 f = bsdf.f(hp.wo, wi.dir);
		if(!zero(f) && light.visible(geometry, wi))
			return Le + f * Li * fmaxf(0.0f, dot(wi.dir, bsdf.N));
	}
	return Le;
}

inline __device__ float3 estimateDirectRadiance(const PhotonMapperParams& params,
	RNG* rng, const Geometry& geometry, const Shader* shaders,
	const Light& light, const HitPoint& hp)
{
	const Shader& shader = shaders[getSafeID(geometry.shaders[hp.triangleID])];
	const BSDF&   bsdf   = shader.getBSDF(geometry, hp.triangleID, hp.u, hp.v);

	Ray wi;
	float3 L = make_float3(0.0f);
	for(unsigned int i=0; i<light.samples; i++) {
		float3 Ls = make_float3(0.0f);

#if 1
		// Sample light
		wi = Ray(hp.position);

		float lightPdf;
		const float3 Li = light.sampleL(rng, wi, lightPdf);
		wi.offset();

		if(lightPdf > 0.0f && !zero(Li)) {
			const float3 f = bsdf.f(hp.wo, wi.dir);
			if(!zero(f) && light.visible(geometry, wi)) {
				const float bsdfPdf = bsdf.pdf(hp.wo, wi.dir);
				const float weight  = powerHeuristic(1, lightPdf, 1, bsdfPdf);
				Ls = Ls + f * Li * (fmaxf(0.0f, dot(wi.dir, bsdf.N)));// * (weight / emitter.pdf));
				//Ls = Ls + f * Li * fmaxf(0.0f, dot(wi.dir, bsdf.N)) * weight / emitter.pdf;
			}
		}
#endif
#if 0

		// Sample BSDF
		wi = Ray(hp.position);

		float bsdfPdf;
		const float3 f = bsdf.samplef(rng, hp.wo, wi.dir, bsdfPdf);
		wi.offset();

		if(bsdfPdf > 0.0f && !zero(f)) {
			const float weight = powerHeuristic(1, bsdfPdf, 1, emitter.pdf);
			wi.t = Infinity;
			if(emitter.visible(geometry, wi)) {
				const float3 Li = emitter.power;
				Ls = Ls + f * Li * fmaxf(0.0f, dot(wi.dir, bsdf.N));
				//Ls = Ls + f *  Li * fmaxf(0.0f, dot(wi.dir, bsdf.N)) * weight / bsdfPdf;
			}
		}
#endif
		L = L + Ls;
	}

	return make_float3(
		L.x / light.samples,
		L.y / light.samples,
		L.z / light.samples);
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

	float3 Li = make_float3(0.0f);
	for(unsigned int i=0; i<params.numLights; i++) {
		const Light& light = lights[i];
		if(light.isDeltaLight())
			Li = Li + estimateDirectRadianceDelta(params, geometry, shaders, light, hp);
		else
			Li = Li + estimateDirectRadiance(params, &rng, geometry, shaders, light, hp);
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

	//cudaGeneratePhotons<<<gridSize, blockSize>>>(rng, geometry, 
	//	params.numEmitters, emitters, params.numPhotons, photons);

	blockSize = dim3(64);
	gridSize  = make_grid(blockSize, dim3(params.numHitPoints));
	cudaRenderDirect<<<gridSize, blockSize>>>(params, rng, geometry, shaders.items, lights.items, hitpoints);

	//gridSize = make_grid(blockSize, dim3(params.numHitPoints));
	//cudaDebugPhotons<<<gridSize, blockSize>>>(params.numHitPoints, hitpoints, params.numPhotons, photons);
}