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
	Ray() : t(Infinity), id(0), weight(1.0f)
	{ }

	__host__ __device__ 
	Ray(const float3& p, const float3& d, const float w=1.0f) : t(Infinity), id(0), weight(w)
	{ }

	__host__ __device__
	inline bool intersect(const Primitive3& triangle, float2& p, float& t) const
	{
		float3 e1 = triangle.v2 - triangle.v1;
		float3 e2 = triangle.v3 - triangle.v1;

		float3 P  = cross(dir, e2);
		float det = dot(e1, P);

		if(det < Epsilon)
			return false;
		float invdet = 1.0f / det;

		float3 T = pos - triangle.v1;
		p.x      = dot(T, P) * invdet;
		if(p.x < 0.0f || p.x > 1.0f)
			return false;

		float3 Q = cross(T, e1);
		p.y      = dot(dir, Q) * invdet;
		if(p.y < 0.0f || p.x + p.y > 1.0f)
			return false;

		t = dot(e2, Q) * invdet;
		return true;
	}

	__host__ __device__
	inline bool intersect(const float2& slab, const int axis, float2& range) const
	{
		float2 dist;
		switch(axis) {
		case 0: dist = (slab - pos.x) / dir.x; break;
		case 1: dist = (slab - pos.y) / dir.y; break;
		case 2: dist = (slab - pos.z) / dir.z; break;
		}

		range.x = max(min(dist.x, dist.y), range.x);
		range.y = min(max(dist.x, dist.y), range.y);
		return (range.x <= range.y) && (range.x <= t);
	}

	float3 pos;
	float3 dir;
	float2 uv;
	float t;
	float weight;
	unsigned int id;
};

} // Aurora