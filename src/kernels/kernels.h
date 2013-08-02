/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <core/geometry.h>

#include <util/math.h>
#include <util/ray.h>
#include <util/camera.h>
#include <util/transform.h>
#include <util/primitive.h>

#ifdef __CUDACC__
inline dim3 make_grid(const dim3& blockSize, const dim3& domainSize)
{
	dim3 gridSize;
	gridSize.x = (domainSize.x / blockSize.x) + (domainSize.x % blockSize.x ? 1 : 0);
	gridSize.y = (domainSize.y / blockSize.y) + (domainSize.y % blockSize.y ? 1 : 0);
	gridSize.z = (domainSize.z / blockSize.z) + (domainSize.z % blockSize.z ? 1 : 0);
	return gridSize;
}
#endif

// NVCC does not properly support namespaces thus kernel wrapper functions need to be defined outside of Aurora scope.

void cudaGenerateRays(const Aurora::Rect& region, const Aurora::Camera& camera, Aurora::Ray* rays);
void cudaTransform(const Aurora::Geometry& geometry, Aurora::Geometry& dest, const Aurora::Transform* transforms, const unsigned int objectCount);
void cudaRaycast(const unsigned int numRays, const Aurora::Ray* rays, const Aurora::Geometry& geometry, void* pixels);
bool cudaRebuildNMH(Aurora::Geometry& geometry);