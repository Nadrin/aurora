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