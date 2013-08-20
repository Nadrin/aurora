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

__global__ static void cudaRaytraceMonteCarloKernel(const Geometry geometry, const ShadersArray shaders, const LightsArray lights,
	RNG* grng, const unsigned int numRays, Ray* rays, HitPoint* hits)
{
	const unsigned int rayID = blockDim.x * blockIdx.x + threadIdx.x;
	if(rayID >= numRays)
		return;

	RNG rng		  = grng[threadIdx.x];
	Ray ray	      = rays[rayID];
	HitPoint& hit = hits[rayID];

	if(!intersect(geometry, ray, hit))
		return;

	const float3 P = ray.point();

	const unsigned int shaderID = getSafeID(geometry.shaders[hit.triangleID]);
	const Shader shader = shaders[shaderID];
	const BSDF   bsdf   = shader.getBSDF(geometry, hit.triangleID, hit.u, hit.v);
	const float3 Wo     = -ray.dir;

	hit.color = shader.ambientColor;
	for(unsigned int l=0; l<lights.size; l++) {
		const Light& light = lights[l];
		
		float3 Li = make_float3(0.0f);
		for(unsigned int s=0; s<lights[l].samples; s++) {			
			float3 Ls = make_float3(0.0f);
			
			float3 Wi, Le, f;
			float  Lpdf, BSDFpdf, Ldistance, weight;

			// Sample light source
			Le = light.sampleL(&rng, P, Wi, Ldistance, Lpdf);
			if(Lpdf > 0.0f && !zero(Le)) {
				f = bsdf.f(Wo, Wi);

				if(!zero(f) && !intersectAny(geometry, Ray(P, Wi, Ldistance).offset())) {
					if(light.isDeltaLight()) { 
						Ls = Ls + f * Le * fmaxf(0.0f, dot(bsdf.N, Wi)) / Lpdf;
					}
					else {
						BSDFpdf = bsdf.pdf(Wo, Wi);
						weight  = powerHeuristic(1, BSDFpdf, 1, Lpdf);
						Ls = Ls + f * Le * fmaxf(0.0f, dot(bsdf.N, Wi)) * weight / Lpdf;
					}
				}
			}

			// Sample BSDF
			if(!light.isDeltaLight()) {

			}

			Li = Li + Ls;
		}
		if(lights[l].samples > 0)
			hit.color = hit.color + (Li / lights[l].samples);
	}
}

__host__ void cudaRaytraceMonteCarlo(const Geometry& geometry, const ShadersArray& shaders, const LightsArray& lights,
	RNG* rng, const unsigned int numRays, Ray* rays, HitPoint* hits)
{
	dim3 blockSize(256);
	dim3 gridSize = make_grid(blockSize, dim3(numRays));
	cudaRaytraceMonteCarloKernel<<<gridSize, blockSize>>>(geometry, shaders, lights, rng, numRays, rays, hits);
}