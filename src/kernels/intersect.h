/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <data/geometry.h>
#include <util/ray.h>

__device__ bool intersectAny(const Aurora::Geometry& geometry, Aurora::Ray& ray);
__device__ bool intersect(const Aurora::Geometry& geometry, Aurora::Ray& ray, unsigned int& triangleIndex);
