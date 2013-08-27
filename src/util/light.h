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
	float  invarea;
	short  samples;

	float3 position;
	float3 color;
	float3 direction;

	float3 e1, e2;

	__device__ float3 sampleL(RNG* rng, Ray& ray, float& pdf) const;
	__device__ bool   visible(const Geometry& geometry, const Ray& ray) const;

	__device__ float  pdf() const;
	__device__ bool   isDeltaLight() const;

};



typedef Array<Light, DeviceMemory>  LightsArray;

} // Aurora