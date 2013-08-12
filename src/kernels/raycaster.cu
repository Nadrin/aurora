/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <kernels/kernels.h>

using namespace Aurora;

#include <kernels/common.h>
#include <kernels/intersect.h>

texture<float, 2, cudaReadModeElementType> texRef1; 

__global__ static void cudaRaycastKernel(const unsigned int numRays, const Geometry geometry, Ray* rays, float4* pixels)
{
	unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= numRays)
		return;

	float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	unsigned int triangleIndex;
	Ray ray = rays[threadId];
	ray.t   = Infinity;

	if(intersect(geometry, ray, triangleIndex)) {
		Primitive normals;
		normals.readValues(geometry.normals + triangleIndex * Geometry::TriangleParams);

		const float3 L = make_float3(0.0f, 0.5f, 0.5f);
		const float3 N = normalize(bclerp(normals.v1, normals.v2, normals.v3, ray.uv.x, ray.uv.y));
		const float dotNL = dot(N, L);

		color = make_float4(0.2f, 0.2f, 0.2f, 1.0f);
		if(dotNL > 0.0f) {
			color.x += dotNL;
			color.y += dotNL;
			color.z += dotNL;
		}
	}

	color.x = clamp(color.x, 0.0f, 1.0f);
	color.y = clamp(color.y, 0.0f, 1.0f);
	color.z = clamp(color.z, 0.0f, 1.0f);
	pixels[ray.id] = color;
}

void cudaRaycast(const unsigned int numRays, const Geometry& geometry, Ray* rays, void* pixels)
{
	dim3 blockSize(256);
	dim3 gridSize = make_grid(blockSize, dim3(numRays));
	cudaRaycastKernel<<<gridSize, blockSize>>>(numRays, geometry, rays, (float4*)pixels);
}