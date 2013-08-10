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

// NVCC does not properly support namespaces thus kernel wrapper functions need to be defined outside of Aurora scope.

void cudaGenerateRays(const Aurora::Rect& region, const Aurora::Camera& camera, Aurora::Ray* rays);
void cudaTransform(const Aurora::Geometry& geometry, Aurora::Geometry& dest, const Aurora::Transform* transforms, const unsigned int objectCount);
void cudaRaycast(const unsigned int numRays, const Aurora::Geometry& geometry, Aurora::Ray* rays, void* pixels);
bool cudaRebuildNMH(Aurora::Geometry& geometry);