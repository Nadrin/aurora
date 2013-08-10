/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <kernels/kernels.h>

using namespace Aurora;

#include <kernels/common.h>

__global__ static void cudaGenerateRaysKernel(const uint2 size, const Camera camera, Ray* rays)
{
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= size.x || y >= size.y)
		return;

	const unsigned int rayID = y * size.x + x;
	const float2 plane  = make_float2(x / float(size.x) - 0.5f, y / float(size.y) - 0.5f);

	Ray* r    = &rays[rayID];
	r->pos    = camera.position;
	r->t      = Infinity;
	r->weight = 1.0f;
	r->id     = rayID;

	r->dir = normalize(
		  plane.x * camera.right * camera.tanfov.x
		+ plane.y * camera.up * camera.tanfov.y
		+ camera.forward);
}

__host__ void cudaGenerateRays(const Rect& region, const Camera& camera, Ray* rays)
{
	uint2 size = make_uint2(region.right - region.left + 1, region.top - region.bottom + 1);

	dim3 blockSize(32, 16);
	dim3 gridSize = make_grid(blockSize, dim3(size.x, size.y));
	cudaGenerateRaysKernel<<<gridSize, blockSize>>>(size, camera, rays);
}

__global__ static void cudaTransformKernel(const Geometry source, Geometry dest, const Transform* transforms, const unsigned int objectCount)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId > source.count)
		return;

	const Transform* xform = NULL;
	for(unsigned int i=0; i<objectCount; i++) {
		if(transforms[i].offset <= threadId)
			xform = &transforms[i];
	}
	if(!xform) return;

	Primitive sourcePrim, destPrim;

	// Transform vertices
	sourcePrim.readPoints(source.vertices + threadId * Geometry::TriangleParams);
	destPrim.v1 = xform->getPoint(sourcePrim.v1);
	destPrim.v2 = xform->getPoint(sourcePrim.v2);
	destPrim.v3 = xform->getPoint(sourcePrim.v3);
	destPrim.writePoints(dest.vertices + threadId * Geometry::TriangleParams);

	// Transform normals
	sourcePrim.readValues(source.normals + threadId * Geometry::TriangleParams);
	destPrim.v1 = xform->getNormal(sourcePrim.v1);
	destPrim.v2 = xform->getNormal(sourcePrim.v2);
	destPrim.v3 = xform->getNormal(sourcePrim.v3);
	destPrim.writeValues(dest.normals + threadId * Geometry::TriangleParams);
}

__host__ void cudaTransform(const Geometry& source, Geometry& dest, const Transform* transforms, const unsigned int objectCount)
{
	dim3 blockSize(256);
	dim3 gridSize = make_grid(blockSize, dim3(source.count));
	cudaTransformKernel<<<gridSize, blockSize>>>(source, dest, transforms, objectCount);
}