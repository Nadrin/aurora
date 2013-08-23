/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <core/renderer.h>
#include <kernels/kernels.h>

#include <maya/MFnCamera.h>

using namespace Aurora;

void Renderer::generateRays(const MDagPath& camera, const Dim& size, const Rect& region,
	Ray* rays, HitPoint* hit)
{
	MFnCamera dagCamera(camera);
	MPoint camEyePoint   = dagCamera.eyePoint(MSpace::kWorld);
	MVector camDirection = dagCamera.viewDirection(MSpace::kWorld);
	double camAspect     = dagCamera.aspectRatio();

	double camFovX, camFovY;
	dagCamera.getPortFieldOfView(size.width, size.height, camFovX, camFovY);

	Camera rayCamera(
		make_float3((float)camEyePoint.x, (float)camEyePoint.y, (float)camEyePoint.z),
		make_float3((float)camDirection.x, (float)camDirection.y, (float)camDirection.z),
		make_float2((float)camFovX, (float)camFovY), (float)camAspect);

	cudaGenerateRays(region, size.depth, rayCamera, rays, hit);
}

void Renderer::drawPixels(const Dim& size, const Rect& region, const HitPoint* hit, void* pixels)
{
	cudaDrawPixels(size, region, size.depth, hit, pixels);
}

bool Renderer::setupRNG(RNG** rng, const size_t count, const unsigned int seed)
{
	if(gpu::cudaMalloc(rng, count * sizeof(RNG)) != gpu::cudaSuccess)
		return false;

	cudaSetupRNG(*rng, count, seed);
	return true;
}