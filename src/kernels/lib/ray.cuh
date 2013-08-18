/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

inline __device__ bool Ray::intersect(const Primitive3& triangle, float& u, float& v, float& t) const
{
	float3 e1 = triangle.v2 - triangle.v1;
	float3 e2 = triangle.v3 - triangle.v1;

	float3 P  = cross(dir, e2);
	float det = dot(e1, P);

	if(det < Epsilon)
		return false;
	float invdet = 1.0f / det;

	float3 T = pos - triangle.v1;
	u        = dot(T, P) * invdet;
	if(u < 0.0f || u > 1.0f)
		return false;

	float3 Q = cross(T, e1);
	v        = dot(dir, Q) * invdet;
	if(v < 0.0f || u + v > 1.0f)
		return false;
	
	t = dot(e2, Q) * invdet;
	return true;
}

inline __device__ bool Ray::intersect(const float2& slab, const int axis, float2& range) const
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

inline __device__ float3 Ray::point() const
{
	return pos + dir * t;
}
