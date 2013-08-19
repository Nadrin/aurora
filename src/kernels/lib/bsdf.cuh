/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <kernels/lib/common.cuh>
#include <kernels/lib/sampling.cuh>

// Local (shading space) functions

inline __device__ float3 BSDF::lf(const float3& wo, const float3& wi) const
{
	switch(type) {
	case BSDF_Lambert:
		return spectrum * InvPi;
	}
}

inline __device__ float  BSDF::lpdf(const float3& wo, const float3& wi) const
{
	switch(type) {
	case BSDF_Lambert:
		return sameHemisphere(wo, wi) ? absCosTheta(wi) * InvPi : 0.0f;
	}
}

// Global (world space) functions

inline __device__ float3 BSDF::f(const float3& wo, const float3& wi) const
{
	return BSDF::lf(worldToLocal(wo, N, S, T), worldToLocal(wi, N, S, T));
}

inline __device__ float  BSDF::pdf(const float3& wo, const float3& wi) const
{
	return BSDF::lpdf(worldToLocal(wo, N, S, T), worldToLocal(wi, N, S, T));
}

inline __device__ float3 BSDF::samplef(RNG* rng, const float3& wo, float3& wi, float& pdf) const
{
	const float3 lwo = worldToLocal(wo, N, S, T);
	float3 result;
	
	wi = sampleHemisphereCosine(curand_uniform(rng), curand_uniform(rng));
	if(lwo.z < 0.0f)
		wi.z *= -1.0f;

	pdf    = BSDF::lpdf(lwo, wi);
	result = BSDF::f(lwo, wi);

	wi = localToWorld(wi, N, S, T);
	return result;
}