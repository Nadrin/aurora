/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <kernels/kernels.h>

using namespace Aurora;

#include <kernels/common.h>

static __device__ __constant__ float3 constUp;
static __device__ __constant__ float3 constForward;

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
	if(threadId >= source.count)
		return;

	const Transform* xform = NULL;
	for(unsigned int i=0; i<objectCount; i++) {
		if(transforms[i].offset <= threadId)
			xform = &transforms[i];
	}
	if(!xform) return;

	Primitive3 sourcePrim, destPrim;

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

	cudaMemcpy(dest.texcoords, source.texcoords, source.count * Geometry::TriangleUVSize, cudaMemcpyDeviceToDevice);
	cudaMemcpy(dest.shaders, source.shaders, source.count * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
}

__global__ static void cudaGenerateTBKernel(const Geometry geometry)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadId >= geometry.count)
		return;

	Primitive3 normal;
	Primitive3 result;

	normal.readValues(geometry.normals + threadId * Geometry::TriangleParams);

	// Calculate tangents
	result.v1 = cross(normal.v1, constUp);
	result.v2 = cross(normal.v2, constUp);
	result.v3 = cross(normal.v3, constUp);

	if(length(result.v1) < Epsilon)
		result.v1 = cross(normal.v1, constForward);
	if(length(result.v2) < Epsilon)
		result.v2 = cross(normal.v2, constForward);
	if(length(result.v3) < Epsilon)
		result.v3 = cross(normal.v3, constForward);

	result.writeValues(geometry.tangents + threadId * Geometry::TriangleParams);

	// Calculate bitangents
	result.v1 = cross(normal.v1, result.v1);
	result.v2 = cross(normal.v2, result.v2);
	result.v3 = cross(normal.v3, result.v3);

	result.writeValues(geometry.bitangents + threadId * Geometry::TriangleParams);
}

__host__ void cudaGenerateTB(const Geometry& geometry)
{
	const float3 vecUp      = make_float3(0.0f, 1.0f, 0.0f);
	const float3 vecForward = make_float3(0.0f, 0.0f, 1.0f);

	cudaMemcpyToSymbol((const char*)&constUp, &vecUp, sizeof(float3), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol((const char*)&constForward, &vecForward, sizeof(float3), 0, cudaMemcpyHostToDevice);

	dim3 blockSize(256);
	dim3 gridSize = make_grid(blockSize, dim3(geometry.count));
	cudaGenerateTBKernel<<<gridSize, blockSize>>>(geometry);
}

__global__ static void cudaSetupRNGKernel(RNG* state, const size_t count, const unsigned int seed)
{
	const unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(seed, threadId, 0, &state[threadId]);
}

__host__ void cudaSetupRNG(RNG* state, const size_t count, const unsigned int seed)
{
	dim3 blockSize(64);
	dim3 gridSize = make_grid(blockSize, dim3(count));
	cudaSetupRNGKernel<<<gridSize, blockSize>>>(state, count, seed);
}