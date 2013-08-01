/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <util/math.h>
#include <util/transform.h>

namespace Aurora {

class Primitive
{
public:
	__host__ __device__ inline void readPoints(const float* v)
	{
		v1.x = v[0]; v2.x = v[1]; v3.x = v[2];
		v1.y = v[3]; v2.y = v[4]; v3.y = v[5];
		v1.z = v[6]; v2.z = v[7]; v3.z = v[8];
	}

	__host__ __device__ inline void readValues(const float* v)
	{
		v1.x = v[0]; v1.y = v[1]; v1.z = v[2];
		v2.x = v[3]; v2.y = v[4]; v2.z = v[5];
		v3.x = v[6]; v3.y = v[7]; v3.z = v[8];
	}

	__host__ __device__ inline void writePoints(float* v)
	{
		v[0] = v1.x; v[1] = v2.x; v[2] = v3.x;
		v[3] = v1.y; v[4] = v2.y; v[5] = v3.y;
		v[6] = v1.z; v[7] = v2.z; v[8] = v3.z;
	}

	__host__ __device__ inline void writeValues(float* v)
	{
		v[0] = v1.x; v[1] = v1.y; v[2] = v1.z;
		v[3] = v2.x; v[4] = v2.y; v[5] = v2.z;
		v[6] = v3.x; v[7] = v3.y; v[8] = v3.z;
	}

	float3 v1;
	float3 v2;
	float3 v3;
};

class Geometry
{
public:
	enum GeometryAllocMode {
		AllocEmpty = 0,
		AllocDefault,
		AllocStaging,
	};

	float* vertices;
	float* normals;
	unsigned int count;
	GeometryAllocMode mode;
protected:
	__host__ bool resizeDefault(const unsigned int n);
	__host__ bool resizeStaging(const unsigned int n);
	__host__ Geometry convertToDevice() const;
public:
	__host__ void initialize();
	__host__ bool resize(const unsigned int n, GeometryAllocMode allocMode);
	__host__ void free();

	__host__ bool copyToDevice(Geometry& other) const;
	__host__ bool copyToDeviceTransform(Geometry& other, const Transform* transforms, const unsigned int count) const;
	__host__ bool padToEven(const unsigned int n);

	static const size_t TriangleSize   = 9 * sizeof(float);
	static const size_t TriangleParams = 9;
};

} // Aurora