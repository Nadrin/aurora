/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <kernels/kernels.h>

using namespace Aurora;

__global__ static void cudaGenerateRaysKernel(const uint2 size, const Camera camera, Ray* rays)
{
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x > size.x || y > size.y)
		return;

	const unsigned int rayID = y * size.x + x;
	const float2 plane  = make_float2(x / float(size.x) - 0.5f, y / float(size.y) - 0.5f);

	Ray* r = &rays[rayID];
	r->pos = camera.position;
	r->t   = Infinity;
	r->id  = rayID;

	r->dir = normalize(
		  plane.x * camera.right * camera.tanfov.x
		+ plane.y * camera.up * camera.tanfov.y
		+ camera.forward);
}

__host__ void cudaGenerateRays(const Rect& region, const Camera& camera, Ray* rays)
{
	uint2 size = make_uint2(region.right - region.left + 1, region.top - region.bottom + 1);

	dim3 blockSize(32, 32);
	dim3 gridSize = make_grid(blockSize, dim3(size.x, size.y));
	cudaGenerateRaysKernel<<<gridSize, blockSize>>>(size, camera, rays);
}