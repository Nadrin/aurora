/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <util/math.h>

namespace Aurora {

class Transform
{
public:
	__host__ __device__
	Transform() : offset(0), size(0)
	{ }

	__device__ float3 getPoint(const float3& p) const;
	__device__ float3 getNormal(const float3& n) const;

	unsigned int offset;
	unsigned int size;
	float worldMatrix[4][4];
	float normalMatrix[4][4];
};

} // Aurora