/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <kernels/kernels.h>

using namespace Aurora;

#include <kernels/lib/common.cuh>
#include <kernels/lib/intersect.cuh>
#include <kernels/lib/bsdf.cuh>
#include <kernels/lib/shader.cuh>
#include <kernels/lib/light.cuh>
#include <kernels/lib/radiance.cuh>
#include <kernels/lib/stack.cuh>

#define MAX_DEPTH 8

typedef Stack<Ray, MAX_DEPTH*2> RayStack;

inline __device__ bool raytrace(RNG* rng, const Geometry& geometry,
	const ShadersArray& shaders, const LightsArray& lights, RayStack& rs, HitPoint& hp, int& depth)
{
	Ray ray = rs.pop();
	if(!intersect(geometry, ray, hp))
		return;

	hp.position = ray.point();
	hp.wo       = -ray.dir;

	const Shader& shader = shaders[getSafeID(geometry.shaders[hp.triangleID])];
	const BSDF&   bsdf   = shader.getBSDF(geometry, hp.triangleID, hp.u, hp.v);

	float3 Li = shader.emissionColor;
	for(int i=0; i<lights.size; i++) {
		const Light& light = lights[i];
		if(light.isDeltaLight())
			Li = Li + estimateDirectRadianceDelta(geometry, light, shader, bsdf, hp);
		else
			Li = Li + estimateDirectRadiance(rng, geometry, light, shader, bsdf, hp);
	}

	if(shader.reflectivity > 0.0f || shader.translucence > 0.0f)
		depth++;

	if(depth <= MAX_DEPTH && shader.reflectivity > 0.0f) {
		Ray newray(hp.position, reflect(ray.dir, bsdf.N));
		newray.weight = shader.reflectivity;
		newray.eta    = ray.eta;
		ray.weight   *= (1.0f - shader.reflectivity);

		rs.push(newray);
	}
	if(depth <= MAX_DEPTH && shader.translucence > 0.0f) {
		float3 rN  = bsdf.N;
		float etai = ray.eta;
		float etat = shader.refractiveIndex;

		if(dot(rN, hp.wo) < 0.0f) {
			rN   = -rN;
			etai = shader.refractiveIndex;
			etat = 1.0f;
		}

		float3 tdir;
		if(refract(tdir, ray.dir, rN, etai, etat)) {
			Ray newray(hp.position, tdir);
			newray.weight = shader.translucence;
			newray.eta    = etat;
			ray.weight   *= (1.0f - shader.translucence);

			rs.push(newray);
		}
	}

	hp.color = hp.color + (Li * ray.weight);
}

__global__ static void cudaRaytraceKernel(const Geometry geometry, const ShadersArray shaders, const LightsArray lights,
	RNG* grng, const unsigned int numRays, Ray* rays, HitPoint* hits)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= numRays)
		return;
	
	RNG rng      = grng[threadId];
	Ray& ray     = rays[threadId];
	HitPoint& hp = hits[threadId];

	RayStack rs;
	rs.push(ray);

	int depth=0;
	while(depth <= MAX_DEPTH && rs.size > 0)
		raytrace(&rng, geometry, shaders, lights, rs, hp, depth);
}

__host__ void cudaRaytraceMonteCarlo(const Geometry& geometry, const ShadersArray& shaders, const LightsArray& lights,
	RNG* rng, const unsigned int numRays, Ray* rays, HitPoint* hits)
{
	dim3 blockSize(64);
	dim3 gridSize = make_grid(blockSize, dim3(numRays));
	cudaRaytraceKernel<<<gridSize, blockSize>>>(geometry, shaders, lights, rng, numRays, rays, hits);
}