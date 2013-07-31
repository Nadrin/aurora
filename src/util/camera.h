/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <util/ray.h>

namespace Aurora {

class Camera
{
public:
	__host__ __device__
	Camera(const float3& position, const float3& direction, const float2& fov, const float aspect)
	{
		const float3 _up = make_float3(0.0f, 1.0f, 0.0f);

		forward  = normalize(direction);
		right    = cross(forward, _up);
		up       = cross(right, forward);
		tanfov.x = tanf(fov.x * 0.5f);
		tanfov.y = tanf(fov.y * 0.5f);

		this->aspect   = aspect;
		this->position = position;
	}

	float3 position;
	float3 forward;
	float3 right;
	float3 up;
	float2 tanfov;
	float  aspect;
};

} // Aurora