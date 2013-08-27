/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

inline __device__ float3 Emitter::sampleL(RNG* rng, const Geometry& geometry, Ray& ray) const
{
	float u, v;
	sampleTriangle(curand_uniform(rng), curand_uniform(rng), u, v);
	const float3 P = getPosition(geometry, triangleID, u, v);
	//const float3 P = getPosition(geometry, triangleID, 0.5f, 0.5f);

	ray.dir  = P - ray.pos;
	ray.t    = length(ray.dir);
	ray.dir  = ray.dir / ray.t;
	return power;
}

inline __device__ bool Emitter::visible(const Geometry& geometry, const Ray& ray) const
{
	unsigned int intersectId;
	if(intersectAny(geometry, ray, intersectId) && intersectId == triangleID)
		return true;
	return false;
}