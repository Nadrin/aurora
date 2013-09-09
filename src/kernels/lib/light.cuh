/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

inline __device__ bool Light::isDeltaLight() const
{
	return type != Light::AreaLight;
}

inline __device__ bool Light::isAmbientLight() const
{
	return type == Light::AmbientLight;
}

inline __device__ float3 Light::L(const float3& wi) const
{
	// Area light only
	return (color * intensity) * fmaxf(0.0f, dot(direction, wi));
}

inline __device__ float3 Light::sampleL(RNG* rng, Ray& ray, float& pdf) const
{
	switch(type) {
	case AmbientLight:
		ray.dir = position - ray.pos;
		ray.t   = length(ray.dir);
		ray.dir = ray.dir / ray.t;
		pdf     = 1.0f;
		return (color * intensity) * (1.0f - ambient);
	case PointLight:
		ray.dir = position - ray.pos;
		ray.t   = length(ray.dir);
		ray.dir = ray.dir / ray.t;
		pdf     = 1.0f;
		return (color * intensity) / (ray.t*ray.t);
	case DirectionalLight:
		ray.dir = -direction;
		ray.t   = Infinity;
		pdf     = 1.0f;
		return color * intensity;
	case AreaLight:
		{
			const float3 P = position 
				+ sampleRectangle(e1, e2, curand_uniform(rng), curand_uniform(rng));
			ray.dir = P - ray.pos;
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

inline __device__ float Light::pdf(const Ray& ray) const
{
	switch(type) {
	case AmbientLight:
	case PointLight:
	case DirectionalLight:
		return 0.0f;
	case AreaLight:
		return 1.0f / (fmaxf(0.0f, dot(direction, -ray.dir)) * area);
	}
}

inline __device__ float Light::power(const Geometry& geometry) const
{
	switch(type) {
	case AmbientLight:
		return 4.0f * Pi * luminosity(color) * intensity * (1.0f - ambient);
	case PointLight:
		return 4.0f * Pi * luminosity(color) * intensity;
	case DirectionalLight:
		{
			const float R = getBoundingSphereRadius(geometry);
			return luminosity(color) * intensity * Pi * R * R;
		}
	case AreaLight:
		return luminosity(color) * intensity * area * Pi;
	}
}

inline __device__ Photon Light::emitPhoton(RNG* rng, const Geometry& geometry) const
{
	Photon p;
	switch(type) {
	case AmbientLight:
		p.pos    = position;
		p.wi     = sampleSphere(curand_uniform(rng), curand_uniform(rng));
		p.energy = color * intensity * (1.0f - ambient);
		break;
	case PointLight:
		p.pos    = position;
		p.wi     = sampleSphere(curand_uniform(rng), curand_uniform(rng));
		p.energy = color * intensity;
		break;
	case DirectionalLight:
		// TODO: Implement
		break;
	case AreaLight:
		p.pos	 = position + sampleRectangle(e1, e2, curand_uniform(rng), curand_uniform(rng));
		p.wi	 = direction;
		p.energy = color * intensity;
		break;
	}
	return p;
}