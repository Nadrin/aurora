/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

inline __device__ bool Light::isDeltaLight() const
{
	return type != Light::AreaLight;
}

inline __device__ float3 Light::sampleL(RNG* rng, const float3& P, float3& wi, float& d, float& pdf) const
{
	switch(type) {
	case PointLight:
		d   = distance(position, P);
		wi  = (position - P) / d;
		pdf = 1.0f;
		return color * intensity;
		//return make_float3(intensity / (d*d));
	}
}

inline __device__ float  Light::pdf(const float3& P, const float3& wi) const
{
	switch(type) {
	case PointLight:
		return 0.0f;
	}
}