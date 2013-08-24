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
	Photon(const float3& p, const float3& d, const float3& pwr) : pos(p), dir(d), power(pwr) { }

	__host__ __device__
	Photon(const Ray& ray, const float3& pwr) : pos(ray.pos), dir(ray.dir), power(pwr) { }

	float3 pos;
	float3 dir;
	float3 power;
};

} // Aurora