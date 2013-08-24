/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

inline __device__ void Primitive2::readPoints(const float* v)
{
	v1.x = v[0]; v2.x = v[1]; v3.x = v[2];
	v1.y = v[3]; v2.y = v[4]; v3.y = v[5];
}

inline __device__ void Primitive2::readValues(const float* v)
{
	v1.x = v[0]; v1.y = v[1];
	v2.x = v[3]; v2.y = v[4];
	v3.x = v[6]; v3.y = v[7];
}

inline __device__ void Primitive2::writePoints(float* v)
{
	v[0] = v1.x; v[1] = v2.x; v[2] = v3.x;
	v[3] = v1.y; v[4] = v2.y; v[5] = v3.y;
}

inline __device__ void Primitive2::writeValues(float* v)
{
	v[0] = v1.x; v[1] = v1.y;
	v[3] = v2.x; v[4] = v2.y;
	v[6] = v3.x; v[7] = v3.y;
}

inline __device__ float Primitive2::area() const
{
	return 0.5f * fabsf(v1.x * (v2.y - v3.y) + v2.x * (v3.y - v1.y) + v3.x * (v1.y - v2.y));
}

inline __device__ float2 Primitive2::centroid() const
{
	return make_float2(
		(v1.x + v2.x + v3.x) / 3.0f,
		(v1.y + v2.y + v3.y) / 3.0f);
}

inline __device__ void Primitive3::readPoints(const float* v)
{
	v1.x = v[0]; v2.x = v[1]; v3.x = v[2];
	v1.y = v[3]; v2.y = v[4]; v3.y = v[5];
	v1.z = v[6]; v2.z = v[7]; v3.z = v[8];
}

inline __device__ void Primitive3::readValues(const float* v)
{
	v1.x = v[0]; v1.y = v[1]; v1.z = v[2];
	v2.x = v[3]; v2.y = v[4]; v2.z = v[5];
	v3.x = v[6]; v3.y = v[7]; v3.z = v[8];
}

inline __device__ void Primitive3::writePoints(float* v)
{
	v[0] = v1.x; v[1] = v2.x; v[2] = v3.x;
	v[3] = v1.y; v[4] = v2.y; v[5] = v3.y;
	v[6] = v1.z; v[7] = v2.z; v[8] = v3.z;
}

inline __device__ void Primitive3::writeValues(float* v)
{
	v[0] = v1.x; v[1] = v1.y; v[2] = v1.z;
	v[3] = v2.x; v[4] = v2.y; v[5] = v2.z;
	v[6] = v3.x; v[7] = v3.y; v[8] = v3.z;
}

inline __device__ float Primitive3::area() const
{
	const float3 va = v1 - v2;
	const float3 vb = v1 - v3;
	const float3 vc = v2 - v3;

	const float a = dot(va, va);
	const float b = dot(vb, vb);
	const float c = dot(vc, vc);

	return sqrtf((2*a*b + 2*b*c + 2*c*a - a*a - b*b - c*c) / 16.0f);
}

inline __device__ float3 Primitive3::centroid() const
{
	return make_float3(
		(v1.x + v2.x + v3.x) / 3.0f,
		(v1.y + v2.y + v3.y) / 3.0f,
		(v1.z + v2.z + v3.z) / 3.0f);
}