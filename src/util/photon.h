/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <util/ray.h>

namespace Aurora {

class Photon
{
public:
	__host__ __device__
	Photon() : weight(1.0f) { }
	
	__host__ __device__
	Photon(const float3& p, const float3& d, const float3& e)
		: pos(p), wi(d), energy(e), weight(1.0f) { }

	__host__ __device__
	Photon(const Ray& ray, const float3& e)
		: pos(ray.pos), wi(ray.dir), energy(e), weight(1.0f) { }

	float3 pos;
	float3 wi;
	float3 energy;
	float  weight;
};

} // Aurora