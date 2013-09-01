/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#ifndef __CUDACC__
#include <maya/MColor.h>
#include <maya/MVector.h>
#include <maya/MMatrix.h>
#include <maya/MFloatVector.h>
#endif

namespace Aurora {

static const float __device__ Infinity   = 1000000.0f;
static const float __device__ Epsilon    = 0.0000001f;
static const float __device__ Pi         = 3.1415926536f;
static const float __device__ TwoPi      = 6.2831853072f;
static const float __device__ InvPi      = 0.3183098862f;
static const float __device__ Radian     = 0.0174532925f;

#ifndef __CUDACC__
typedef gpu::int2 int2;
typedef gpu::int3 int3;
typedef gpu::int4 int4;

typedef gpu::uint2 uint2;
typedef gpu::uint3 unit3;
typedef gpu::uint4 unit4;

typedef gpu::float2 float2;
typedef gpu::float3 float3;
typedef gpu::float4 float4;

template <typename T> inline T min(const T a, const T b)
{ return a < b ? a : b; }
template <typename T> inline T max(const T a, const T b)
{ return a > b ? a : b; }

inline float rsqrtf(const float x)
{ return 1.0f / sqrtf(x); }

inline float2 make_float2(const float x, const float y)
{ return gpu::make_float2(x, y); }
inline float3 make_float3(const float x, const float y, const float z)
{ return gpu::make_float3(x, y, z); }
inline float4 make_float4(const float x, const float y, const float z, const float w)
{ return gpu::make_float4(x, y, z, w); }

inline float3 make_float3(const MColor& color)
{ return gpu::make_float3(color.r, color.g, color.b); }
inline float4 make_float4(const MColor& color)
{ return gpu::make_float4(color.r, color.g, color.b, color.a); }

inline float3 make_float3(const MVector& vector)
{ return gpu::make_float3(float(vector.x), float(vector.y), float(vector.z)); }
inline float3 make_float3(const MFloatVector& vector)
{ return gpu::make_float3(vector.x, vector.y, vector.z); }

inline float3 operator*(const MMatrix& matrix, const float3& v)
{
	float m[4][4];
	float3 result;

	matrix.get(m);
	result.x = m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0];
	result.y = m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1];
	result.z = m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z + m[3][2];
	return result;
}
#endif

// Check for vector zero
inline __host__ __device__ bool zero(const float2& v)
{ return v.x < Epsilon && v.y < Epsilon; }
inline __host__ __device__ bool zero(const float3& v)
{ return v.x < Epsilon && v.y < Epsilon && v.z < Epsilon; }
inline __host__ __device__ bool zero(const float4& v)
{ return v.x < Epsilon && v.y < Epsilon && v.z < Epsilon && v.z < Epsilon; }

// Unary operators
inline __host__ __device__ float2 operator-(const float2& v)
{ return make_float2(-v.x, -v.y); }
inline __host__ __device__ float3 operator-(const float3& v)
{ return make_float3(-v.x, -v.y, -v.z); }
inline __host__ __device__ float4 operator-(const float4& v)
{ return make_float4(-v.x, -v.y, -v.z, -v.w); }

// Addition
inline __host__ __device__ float2 operator+(const float2& a, const float b)
{ return make_float2(a.x + b, a.y + b); }
inline __host__ __device__ float2 operator+(const float a, const float2& b)
{ return make_float2(a + b.x, a + b.y); }
inline __host__ __device__ float2 operator+(const float2& a, const float2& b)
{ return make_float2(a.x + b.x, a.y + b.y); }

inline __host__ __device__ float3 operator+(const float3& a, const float b)
{ return make_float3(a.x + b, a.y + b, a.z + b); }
inline __host__ __device__ float3 operator+(const float a, const float3& b)
{ return make_float3(a + b.x, a + b.y, a + b.z); }
inline __host__ __device__ float3 operator+(const float3& a, const float3& b)
{ return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }

inline __host__ __device__ float4 operator+(const float4& a, const float b)
{ return make_float4(a.x + b, a.y + b, a.z + b, a.w + b); }
inline __host__ __device__ float4 operator+(const float a, const float4& b)
{ return make_float4(a + b.x, a + b.y, a + b.z, a + b.w); }
inline __host__ __device__ float4 operator+(const float4& a, const float4& b)
{ return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }

// Subtraction
inline __host__ __device__ float2 operator-(const float2& a, const float b)
{ return make_float2(a.x - b, a.y - b); }
inline __host__ __device__ float2 operator-(const float a, const float2& b)
{ return make_float2(a - b.x, a - b.y); }
inline __host__ __device__ float2 operator-(const float2& a, const float2& b)
{ return make_float2(a.x - b.x, a.y - b.y); }

inline __host__ __device__ float3 operator-(const float3& a, const float b)
{ return make_float3(a.x - b, a.y - b, a.z - b); }
inline __host__ __device__ float3 operator-(const float a, const float3& b)
{ return make_float3(a - b.x, a - b.y, a - b.z); }
inline __host__ __device__ float3 operator-(const float3& a, const float3& b)
{ return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }

inline __host__ __device__ float4 operator-(const float4& a, const float b)
{ return make_float4(a.x - b, a.y - b, a.z - b, a.w - b); }
inline __host__ __device__ float4 operator-(const float a, const float4& b)
{ return make_float4(a - b.x, a - b.y, a - b.z, a - b.w); }
inline __host__ __device__ float4 operator-(const float4& a, const float4& b)
{ return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }

// Multiplication
inline __host__ __device__ float2 operator*(const float2& a, const float b)
{ return make_float2(a.x * b, a.y * b); }
inline __host__ __device__ float2 operator*(const float a, const float2& b)
{ return make_float2(a * b.x, a * b.y); }
inline __host__ __device__ float2 operator*(const float2& a, const float2& b)
{ return make_float2(a.x * b.x, a.y * b.y); }

inline __host__ __device__ float3 operator*(const float3& a, const float b)
{ return make_float3(a.x * b, a.y * b, a.z * b); }
inline __host__ __device__ float3 operator*(const float a, const float3& b)
{ return make_float3(a * b.x, a * b.y, a * b.z); }
inline __host__ __device__ float3 operator*(const float3& a, const float3& b)
{ return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }

inline __host__ __device__ float4 operator*(const float4& a, const float b)
{ return make_float4(a.x * b, a.y * b, a.z * b, a.w * b); }
inline __host__ __device__ float4 operator*(const float a, const float4& b)
{ return make_float4(a * b.x, a * b.y, a * b.z, a * b.w); }
inline __host__ __device__ float4 operator*(const float4& a, const float4& b)
{ return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }

// Division
inline __host__ __device__ float2 operator/(const float2& a, const float b)
{ return make_float2(a.x / b, a.y / b); }
inline __host__ __device__ float2 operator/(const float a, const float2& b)
{ return make_float2(a / b.x, a / b.y); }
inline __host__ __device__ float2 operator/(const float2& a, const float2& b)
{ return make_float2(a.x / b.x, a.y / b.y); }

inline __host__ __device__ float3 operator/(const float3& a, const float b)
{ return make_float3(a.x / b, a.y / b, a.z / b); }
inline __host__ __device__ float3 operator/(const float a, const float3& b)
{ return make_float3(a / b.x, a / b.y, a / b.z); }
inline __host__ __device__ float3 operator/(const float3& a, const float3& b)
{ return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }

inline __host__ __device__ float4 operator/(const float4& a, const float b)
{ return make_float4(a.x / b, a.y / b, a.z / b, a.w / b); }
inline __host__ __device__ float4 operator/(const float a, const float4& b)
{ return make_float4(a / b.x, a / b.y, a / b.z, a / b.w); }
inline __host__ __device__ float4 operator/(const float4& a, const float4& b)
{ return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); }

// Dot product
inline __host__ __device__ float dot(const float2& a, const float2& b)
{ return a.x * b.x + a.y * b.y; }
inline __host__ __device__ float dot(const float3& a, const float3& b)
{ return a.x * b.x + a.y * b.y + a.z * b.z; }
inline __host__ __device__ float dot(const float4& a, const float4& b)
{ return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

// Length
template <typename T> inline __host__ __device__ float length(const T& v)
{ return sqrtf(dot(v, v)); }

template <typename T> inline __host__ __device__ float lengthSq(const T& v)
{ return dot(v, v); }

// Distance
template <typename T> inline __host__ __device__ float distance(const T& a, const T& b)
{ return length(a - b); }

template <typename T> inline __host__ __device__ float distanceSq(const T& a, const T& b)
{ return lengthSq(a - b); }

// Normalize
template <typename T> inline __host__ __device__ T normalize(const T& v)
{ return v * rsqrtf(dot(v, v)); }

// Lerp
template <typename T> inline __host__ __device__ T lerp(const T& a, const T& b, const float t)
{ return a + t*(b - a); }

// Clamp
template <typename T> inline __host__ __device__ T clamp(const T& v, const T& a, const T& b)
{ return max(min(v, b), a); }

// Barycentric interpolate
template <typename T> inline __host__ __device__ T bclerp(const T& a, const T& b, const T& c, 
	const float u, const float v)
{ return (1.0f - u - v) * a + u * b + v * c; }

// Cross product
inline __host__ __device__ float3 cross(const float3& a, const float3& b)
{ return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); }

// Reflect
inline __host__ __device__ float3 reflect(const float3& i, const float3& n)
{ return i - 2.0f * n * dot(n, i); }

// Refract
inline __host__ __device__ bool refract(float3& r, const float3& i, const float3& n,
	const float eta1, const float eta2)
{
	const float eta  = eta1 / eta2;
	const float cosi = dot(n, i);
	const float sint = eta * eta * (1.0f - cosi * cosi);

	if(sint > 1.0f)
		return false;
	r = eta * i - (eta + sqrtf(1.0f - sint)) * n;
	return true;
}

// Compute RGB color luminosity
inline __host__ __device__ float luminosity(const float3& color)
{ return 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z; }

// Integer log2
inline __host__ __device__ unsigned int log2i(unsigned int x)
{
	unsigned int ret = (unsigned int)(-1);
	while(x) {
		x >>= 1;
		ret++;
	}
	return ret;
}

// Integer exp2
inline __host__ __device__ unsigned int exp2i(unsigned int x)
{
	if(x==0) return 1;

	unsigned int ret = 2;
	while(--x) 
		ret *= 2;
	return ret;
}

// Generic value swap
template <typename T> inline __host__ __device__ void swap(T& a, T& b)
{
	T temp = a;
	a = b;
	b = temp;
}

// Rectangular region helper class
class Rect 
{
public:
	__host__ 
	Rect() : left(0), right(0), top(0), bottom(0) { }

	__host__ __device__ 
	Rect(unsigned int l, unsigned int r, unsigned int b, unsigned int t)
		: left(l), right(r), top(t), bottom(b) { }

	unsigned int left;
	unsigned int right;
	unsigned int bottom;
	unsigned int top;
};

// Dimension helper class
class Dim
{
public:
	__host__
	Dim() : width(0), height(0), depth(0) { }

	__host__ __device__
	Dim(unsigned int w, unsigned int h=1, unsigned int d=1)
		: width(w), height(h), depth(d) { }

	unsigned int width;
	unsigned int height;
	unsigned int depth;
};

} // Aurora