/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

// Common functions
inline __device__ float3 make_float3(const float v)
{ return make_float3(v, v, v);    }

inline __device__ float4 make_float4(const float v)
{ return make_float4(v, v, v, v); }

inline __device__ float4 make_float4(const float3& xyz, const float w=0.0f)
{ return make_float4(xyz.x, xyz.y, xyz.z, w); }

// Kernel execution helpers
inline __host__ dim3 make_grid(const dim3& blockSize, const dim3& domainSize)
{
	dim3 gridSize;
	gridSize.x = (domainSize.x / blockSize.x) + (domainSize.x % blockSize.x ? 1 : 0);
	gridSize.y = (domainSize.y / blockSize.y) + (domainSize.y % blockSize.y ? 1 : 0);
	gridSize.z = (domainSize.z / blockSize.z) + (domainSize.z % blockSize.z ? 1 : 0);
	return gridSize;
}

// Shading space math
inline __device__ float3 getPosition(const Geometry& geometry, const unsigned int index,
	const float u, const float v)
{
	Primitive3 buffer;
	buffer.readPoints(geometry.vertices + index * Geometry::TriangleParams);
	return bclerp(buffer.v1, buffer.v2, buffer.v3, u, v);
}

inline __device__ float3 getNormal(const Geometry& geometry, const unsigned int index,
	const float u, const float v)
{
	Primitive3 buffer;
	buffer.readValues(geometry.normals + index * Geometry::TriangleParams);
	return normalize(bclerp(buffer.v1, buffer.v2, buffer.v3, u, v));
}

inline __device__ void getBasisVectors(const Geometry& geometry, const unsigned int index,
	const float u, const float v, float3& N, float3& S, float3& T)
{
	Primitive3 buffer;
	
	buffer.readValues(geometry.normals + index * Geometry::TriangleParams);
	N = normalize(bclerp(buffer.v1, buffer.v2, buffer.v3, u, v));

	buffer.readValues(geometry.tangents + index * Geometry::TriangleParams);
	S = normalize(bclerp(buffer.v1, buffer.v2, buffer.v3, u, v));

	buffer.readValues(geometry.bitangents + index * Geometry::TriangleParams);
	T = normalize(bclerp(buffer.v1, buffer.v2, buffer.v3, u, v));
}

inline __device__ float3 worldToLocal(const float3& v, const float3& N, const float3& S, const float3& T)
{
	return make_float3(
		v.x * S.x + v.y * S.y + v.z * S.z,
		v.x * T.x + v.y * T.y + v.z * T.z,
		v.x * N.x + v.y * N.y + v.z * N.z);
}

inline __device__ float3 localToWorld(const float3& v, const float3& N, const float3& S, const float3& T)
{
	return make_float3(
		S.x * v.x + T.x * v.y + N.x * v.z,
		S.y * v.x + T.y * v.y + N.y * v.z,
		S.z * v.x + T.z * v.y + N.z * v.z);
}

inline __device__ float cosTheta(const float3& v)
{ return v.z; }

inline __device__ float absCosTheta(const float3& v)
{ return fabsf(v.z); }

inline __device__ float sinThetaSq(const float3& v)
{ return fmaxf(0.0f, 1.0f - v.z*v.z); }

inline __device__ float sinTheta(const float3& v)
{ return sqrtf(sinThetaSq(v)); }

inline __device__ float cosPhi(const float3& v)
{
	const float st = sinTheta(v);
	if(st == 0.0f) return 1.0f;
	return clamp(v.x / st, -1.0f, 1.0f);
}

inline __device__ float sinPhi(const float3& v)
{
	const float st = sinTheta(v);
	if(st == 0.0f) return 0.0f;
	return clamp(v.y / st, -1.0f, 1.0f);
}

inline __device__ bool sameHemisphere(const float3& a, const float3& b)
{ return a.z * b.z > 0.0f; }
