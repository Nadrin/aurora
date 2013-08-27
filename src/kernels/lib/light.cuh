/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

inline __device__ bool Light::isDeltaLight() const
{
	return type != Light::AreaLight;
}

inline __device__ float3 Light::L(const float3& wi) const
{
	// Area light only
	return (color * intensity) * fmaxf(0.0f, dot(direction, wi));
}

inline __device__ float3 Light::sampleL(RNG* rng, Ray& ray, float& pdf) const
{
	switch(type) {
	case PointLight:
		ray.dir = position - ray.pos;
		ray.t   = length(ray.dir);
		ray.dir = ray.dir / ray.t;
		pdf     = 1.0f;
		return (color * intensity);// / ray.t*ray.t;
	case DirectionalLight:
		ray.dir = -direction;
		ray.t   = Infinity;
		pdf     = 1.0f;
		return color * intensity;
	case AreaLight:
		{
			const float u1 = curand_uniform(rng);
			const float u2 = curand_uniform(rng);
			const float3 P = position + (e1 * u1) + (e2 * u2);
			ray.dir        = P - ray.pos;
		}
		ray.t   = length(ray.dir);
		ray.dir = ray.dir / ray.t;
		pdf     = Light::pdf(ray);
		return (color * intensity) * fmaxf(0.0f, dot(direction, -ray.dir));
	}
}

inline __device__ bool Light::visible(const Geometry& geometry, const Ray& ray) const
{
	return !intersectAny(geometry, ray);
}

inline __device__ float  Light::pdf(const Ray& ray) const
{
	switch(type) {
	case PointLight:
	case DirectionalLight:
		return 0.0f;
	case AreaLight:
		return 1.0f / (fmaxf(0.0f, dot(direction, -ray.dir)) * area);
	}
}