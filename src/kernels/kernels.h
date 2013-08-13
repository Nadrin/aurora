/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <data/geometry.h>

#include <util/math.h>
#include <util/ray.h>
#include <util/camera.h>
#include <util/transform.h>
#include <util/primitive.h>

#include <util/shader.h>
#include <util/light.h>

// NVCC does not properly support namespaces thus kernel wrapper functions need to be defined outside of Aurora scope.

// Common utility kernels
void cudaGenerateRays(const Aurora::Rect& region, const Aurora::Camera& camera, Aurora::Ray* rays);
void cudaTransform(const Aurora::Geometry& geometry, Aurora::Geometry& dest, const Aurora::Transform* transforms, const unsigned int objectCount);
void cudaGenerateTB(const Aurora::Geometry& geometry);

// NMH construction
bool cudaRebuildNMH(Aurora::Geometry& geometry);

// Rendering
void cudaRaycast(const Aurora::Geometry& geometry, const Aurora::ShadersArray& shaders,
	const unsigned int numRays, Aurora::Ray* rays, void* pixels);