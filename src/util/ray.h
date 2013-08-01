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

		float3 T = pos - triangle.v1;
		p.x      = dot(T, P);
		if(p.x < 0.0f || p.x > det)
			return false;

		float3 Q = cross(T, e1);
		p.y      = dot(dir, Q);
		if(p.y < 0.0f || p.x + p.y > det)
			return false;

		t = dot(e2, Q) / det;
		return true;
	}

	float3 pos;
	float3 dir;
	float t;
	unsigned int id;
};

} // Aurora