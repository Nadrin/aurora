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
	Ray() : t(Infinity), id(0)
	{ }

	__host__ __device__ 
	Ray(const float3& p, const float3& d) : t(Infinity), id(0)
	{ }

	__host__ __device__
	inline bool intersect(const Primitive& triangle, float2& p, float& t) const
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
		float _pos, _rdir;
		switch(axis) {
		case 0: _pos = pos.x; _rdir = 1.0f / dir.x; break;
		case 1: _pos = pos.y; _rdir = 1.0f / dir.y; break;
		case 2: _pos = pos.z; _rdir = 1.0f / dir.z; break;
		}

		const bool reverse = (_rdir < 0.0f);
		
		const float t0 = (slab.x - _pos) * _rdir;
		range.x = (reverse || (t0 <= range.x)) ? range.x : t0;
		range.y = (reverse && (t0 <  range.y)) ? t0 : range.y;

		const float t1 = (slab.y - _pos) * _rdir;
		range.x = (reverse && (t1 >  range.x)) ? t1 : range.x;
		range.y = (reverse || (t1 >= range.y)) ? range.y : t1;

		return ((range.x <= range.y) && (range.x <= t));
	}

	float3 pos;
	float3 dir;
	float2 uv;
	float t;
	unsigned int id;
};

} // Aurora