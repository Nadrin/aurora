/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <util/array.h>
#include <util/ray.h>

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

	__device__ float3 sampleL(RNG* rng, const float3& P, float3& wi, float& d, float& pdf) const;
	__device__ float  pdf(const float3& P, const float3& wi) const;
	__device__ bool   isDeltaLight() const;

};

typedef Array<Light, DeviceMemory>  LightsArray;

} // Aurora