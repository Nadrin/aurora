/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <util/array.h>

namespace Aurora {

class Light
{
public:
	enum LightType {
		AmbientLight,
		PointLight,
		DirectionalLight,
		AreaLight,
	};

	LightType type;

	float  intensity;
	short  samples;
	float3 color;
	float3 position;
	float3 direction;
	float3 scale;

	__device__ float3 getL(const float3& P) const;
};

typedef Array<Light, DeviceMemory>  LightsArray;

} // Aurora