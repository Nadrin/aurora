/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

namespace Aurora {

class Primitive2
{
public:
	__device__ void readPoints(const float* v);
	__device__ void readValues(const float* v);
	__device__ void writePoints(float* v);
	__device__ void writeValues(float* v);

	float2 v1;
	float2 v2;
	float2 v3;
};

class Primitive3
{
public:
	__device__ void readPoints(const float* v);
	__device__ void readValues(const float* v);
	__device__ void writePoints(float* v);
	__device__ void writeValues(float* v);

	float3 v1;
	float3 v2;
	float3 v3;
};

} // Aurora