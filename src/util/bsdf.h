/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

namespace Aurora {

class BSDF
{
public:
	enum BSDFType {
		BSDF_Lambert,
		BSDF_Phong,
	};

	BSDFType type;
	float3	 N, S, T;
	float3   spectrum;

	__device__ float3 f(const float3& wo, const float3& wi) const;
	__device__ float  pdf(const float3& wo, const float3& wi) const;
	__device__ float3 samplef(RNG* rng, const float3& wo, float3& wi, float& pdf) const;
protected:
	__device__ float3 lf(const float3& wo, const float3& wi) const;
	__device__ float  lpdf(const float3& wo, const float3& wi) const;
};

} // Aurora