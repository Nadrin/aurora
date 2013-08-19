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
	const unsigned int numRays, Ray* rays, float4* pixels)
{
	unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= numRays)
		return;

	float3 color = make_float3(0.0f, 0.0f, 0.0f);

	unsigned int triangleIndex;
	Ray ray = rays[threadId];
	ray.t   = Infinity;

	if(intersect(geometry, ray, triangleIndex)) {
		float3 N, T, S;
		getBasisVectors(geometry, triangleIndex, ray.u, ray.v, N, S, T);

		const float3 P = ray.point();

		const unsigned int shaderID = getSafeID(geometry.shaders[triangleIndex]);
		const Shader shader = shaders[shaderID];
		
		color = shader.ambientColor;
		for(unsigned int i=0; i<lights.size; i++) {
			//const float3 L    = worldToLocal(lights[i].getL(P), N, S, T);
			const float3 L = make_float3(0.0f, 0.0f, -1.0f);
			const float dotNL = cosTheta(L);
			if(dotNL > 0.0f)
				color = color + dotNL * shader.diffuse * lights[i].intensity * shader.color * lights[i].color;
		}
	}

	pixels[ray.id] = make_float4(
		clamp(color.x, 0.0f, 1.0f),
		clamp(color.y, 0.0f, 1.0f),
		clamp(color.z, 0.0f, 1.0f),
		1.0f);
}

void cudaRaycast(const Geometry& geometry, const ShadersArray& shaders, const LightsArray& lights,
	const unsigned int numRays, Ray* rays, void* pixels)
{
	dim3 blockSize(256);
	dim3 gridSize = make_grid(blockSize, dim3(numRays));
	cudaRaycastKernel<<<gridSize, blockSize>>>(geometry, shaders, lights, numRays, rays, (float4*)pixels);
}