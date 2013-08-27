/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

namespace Aurora {

class Emitter
{
public:
	float area;
	float pdf;
	float cdf;

	float3 power;
	unsigned int triangleID;

	__device__ float3 sampleL(RNG* rng, const Geometry& geometry, Ray& ray) const;
	__device__ bool   visible(const Geometry& geometry, const Ray& ray) const;
};

} // Aurora