/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <kernels/kernels.h>

using namespace Aurora;

#include <kernels/lib/common.cuh>
#include <kernels/lib/primitive.cuh>
#include <kernels/lib/transform.cuh>

static __device__ __constant__ float3 constUp;
static __device__ __constant__ float3 constForward;

__global__ static void cudaGenerateRaysKernel(const uint2 size, const Camera camera, Ray* rays, HitPoint* hit)
{
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= size.x || y >= size.y)
		return;

	const unsigned int ID = y * size.x + x;
	const float2 pixel = make_float2(float(x) / float(size.x) - 0.5f, float(y) / float(size.y) - 0.5f);

	Ray& ray     = rays[ID];
	HitPoint& hp = hit[ID];

	ray.pos = camera.position;
	ray.t   = Infinity;
	ray.dir = normalize(
		pixel.x * camera.right * camera.tanfov.x +
		pixel.y * camera.up * camera.tanfov.y +
		camera.forward);

	hp.color      = make_float3(0.0f, 0.0f, 0.0f);
	hp.triangleID = -1;
}

__global__ static void cudaGenerateRaysKernelMultisample(const uint2 size, const unsigned short samples, const Camera camera, 
	Ray* rays, HitPoint* hit)
{
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= size.x || y >= size.y)
		return;
	
	const float2 delta  = make_float2(1.0f / float(size.x), 1.0f / float(size.y));
	const short  s      = (short)sqrtf(samples);
	const float2 sdelta = make_float2(delta.x / s, delta.y / s);

	const unsigned int pixelID = y * size.x + x;
	unsigned int rayID = samples * pixelID;

	for(short dy=-s/2; dy<s/2; dy++) {
		for(short dx=-s/2; dx<s/2; dx++, rayID++) {
			const float2 pixel = make_float2(
				x * delta.x - 0.5f + (dx * sdelta.x),
				y * delta.y - 0.5f + (dy * sdelta.y));

			Ray& ray     = rays[rayID];
			HitPoint& hp = hit[rayID];
		
			ray.pos = camera.position;
			ray.t   = Infinity;
			ray.dir = normalize(
				pixel.x * camera.right * camera.tanfov.x +
				pixel.y * camera.up * camera.tanfov.y +
				camera.forward);

			hp.color      = make_float3(0.0f, 0.0f, 0.0f);
			hp.triangleID = -1;
		}
	}
}

__host__ void cudaGenerateRays(const Rect& region, const unsigned short samples, const Camera& camera,
	Ray* rays, HitPoint* hit)
{
	uint2 size = make_uint2(region.right - region.left + 1, region.top - region.bottom + 1);

	dim3 blockSize(32, 16);
	dim3 gridSize = make_grid(blockSize, dim3(size.x, size.y));

	if(samples == 1)
		cudaGenerateRaysKernel<<<gridSize, blockSize>>>(size, camera, rays, hit);
	else
		cudaGenerateRaysKernelMultisample<<<gridSize, blockSize>>>(size, samples, camera, rays, hit);
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
	if(threadId >= count)
		return;

	RNG localState = state[threadId];
	curand_init(seed, threadId, 0, &localState);
	state[threadId] = localState;
}

__host__ void cudaSetupRNG(RNG* state, const size_t count, const unsigned int seed)
{
	dim3 blockSize(64);
	dim3 gridSize = make_grid(blockSize, dim3(count));
	cudaSetupRNGKernel<<<gridSize, blockSize>>>(state, count, seed);
}

__global__ static void cudaDrawPixelsKernel(const Dim size, const Rect region, const unsigned short samples,
	const HitPoint* hit, float4* pixels)
{
	const unsigned int x = region.left   + blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y = region.bottom + blockDim.y * blockIdx.y + threadIdx.y;

	if(x > region.right || y > region.top)
		return;
	
	const float weight = 1.0f / samples;

	const unsigned int pixelID = size.width * y + x;
	const unsigned int rayID   = samples * pixelID;

	float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	for(unsigned short i=0; i<samples; i++) {
		const HitPoint& hp = hit[rayID + i];
		if(hp.triangleID != -1) {
			value.x += hp.color.x * weight;
			value.y += hp.color.y * weight;
			value.z += hp.color.z * weight;
			value.w += weight;
		}
	}
	pixels[pixelID] = value;
}

__host__ void cudaDrawPixels(const Dim& size, const Rect& region, const unsigned short samples,
	const HitPoint* hit, void* pixels)
{
	uint2 pcount = make_uint2(region.right - region.left + 1, region.top - region.bottom + 1);

	dim3 blockSize(32, 16);
	dim3 gridSize = make_grid(blockSize, dim3(pcount.x, pcount.y));
	cudaDrawPixelsKernel<<<gridSize, blockSize>>>(size, region, samples, hit, (float4*)pixels);
}