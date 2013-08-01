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

	__host__ __device__
	inline float3 getPoint(const float3& p) const
	{
		float3 result;
		result.x = worldMatrix[0][0] * p.x + worldMatrix[1][0] * p.y + worldMatrix[2][0] * p.z + worldMatrix[3][0];
		result.y = worldMatrix[0][1] * p.x + worldMatrix[1][1] * p.y + worldMatrix[2][1] * p.z + worldMatrix[3][1];
		result.z = worldMatrix[0][2] * p.x + worldMatrix[1][2] * p.y + worldMatrix[2][2] * p.z + worldMatrix[3][2];
		return result;
	}

	__host__ __device__
	inline float3 getNormal(const float3& n) const
	{
		float3 result;
		result.x = normalMatrix[0][0] * n.x + normalMatrix[1][0] * n.y + normalMatrix[2][0] * n.z + normalMatrix[3][0];
		result.y = normalMatrix[0][1] * n.x + normalMatrix[1][1] * n.y + normalMatrix[2][1] * n.z + normalMatrix[3][1];
		result.z = normalMatrix[0][2] * n.x + normalMatrix[1][2] * n.y + normalMatrix[2][2] * n.z + normalMatrix[3][2];
		return result;
	}

	unsigned int offset;
	unsigned int size;
	float worldMatrix[4][4];
	float normalMatrix[4][4];
};

} // Aurora