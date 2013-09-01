/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <util/math.h>
#include <util/primitive.h>

namespace Aurora {

class Ray 
{
public:
	__host__ __device__
	Ray() : t(Infinity), weight(1.0f)
	{ }

	__host__ __device__
	Ray(const float3& p) : pos(p), t(Infinity), weight(1.0f)
	{ }
	
	__host__ __device__ 
	Ray(const float3& p, const float3& d, const float t=Infinity) : pos(p), dir(d), t(t), weight(1.0f)
	{ }

	__device__
	Ray& offset(const float off=Epsilon);

	__device__
	bool intersect(const Primitive3& triangle, float& u, float& v, float& t) const;

	__device__
	bool intersect(const float2& slab, const int axis, float2& range) const;

	__device__
	float3 point() const;

	float3 pos;
	float3 dir;
	float t;
	float weight;
	float eta;
};

} // Aurora