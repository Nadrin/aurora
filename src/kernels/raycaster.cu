/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <kernels/kernels.h>

using namespace Aurora;

__global__ static void cudaRaycastKernel(const unsigned int numRays, const Ray* rays, const Geometry geometry, float4* pixels)
{
	unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId > numRays)
		return;

	float4 color = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	Ray ray = rays[threadId];

	Primitive triangle;
	Primitive normals;

	unsigned int offset=0, _offset=0;
	float t=Infinity, _t;
	float2 uv;

	for(unsigned int i=0; i<geometry.count; i++) {
		triangle.position(geometry.vertices + _offset);
		if(ray.intersect(triangle, uv, _t) && _t < t) {
			t = _t;
			offset = _offset;
			//break;
		}
		_offset += Geometry::TriangleParams;
	}

	if(t < Infinity) {
		normals.data(geometry.normals + offset);
		const float3 L = make_float3(0.0f, 0.5f, 0.5f);
		const float3 N = normalize(bclerp(normals.v1, normals.v2, normals.v3, uv.x, uv.y));
		const float dotNL = dot(N, L);

		color.x = dotNL;
		color.y = dotNL;
		color.z = dotNL;
	}
	pixels[ray.id] = color;
}

void cudaRaycast(const unsigned int numRays, const Ray* rays, const Geometry& geometry, void* pixels)
{
	dim3 blockSize(256);
	dim3 gridSize = make_grid(blockSize, dim3(numRays));
	cudaRaycastKernel<<<gridSize, blockSize>>>(numRays, rays, geometry, (float4*)pixels);
}