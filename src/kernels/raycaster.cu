/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <kernels/kernels.h>

using namespace Aurora;

#include <kernels/lib/common.cuh>
#include <kernels/lib/intersect.cuh>
#include <kernels/lib/light.cuh>

__global__ static void cudaRaycastKernel(const Geometry geometry, const ShadersArray shaders, const LightsArray lights,
	const unsigned int numRays, Ray* rays, HitPoint* hitpoints)
{
	unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= numRays)
		return;

	Ray&		ray = rays[threadId];
	HitPoint&	hit = hitpoints[threadId];

	if(intersect(geometry, ray, hit)) {
		float3 N, T, S;
		getBasisVectors(geometry, hit.triangleID, hit.u, hit.v, N, S, T);

		const float3 P = ray.point();

		const unsigned int shaderID = getSafeID(geometry.shaders[hit.triangleID]);
		const Shader shader = shaders[shaderID];
		
		hit.color = shader.ambientColor;
		for(unsigned int i=0; i<lights.size; i++) {
			const float3 gL   = normalize(lights[i].position - P);
			const float3 L    = worldToLocal(gL, N, S, T);
			const float dotNL = cosTheta(L);
			if(dotNL > 0.0f)
				hit.color = hit.color + dotNL * shader.diffuse * lights[i].intensity * shader.color * lights[i].color;
		}
	}
}

void cudaRaycast(const Geometry& geometry, const ShadersArray& shaders, const LightsArray& lights,
	const unsigned int numRays, Ray* rays, HitPoint* hitpoints)
{
	dim3 blockSize(256);
	dim3 gridSize = make_grid(blockSize, dim3(numRays));
	cudaRaycastKernel<<<gridSize, blockSize>>>(geometry, shaders, lights, numRays, rays, hitpoints);
}