/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

inline __device__ float3 estimateDirectRadianceDelta(const Geometry& geometry,
	const Light& light, const Shader& shader, const BSDF& bsdf, const HitPoint& hp)
{
	Ray wi(hp.position);

	float pdf;
	const float3 Li = light.sampleL(NULL, wi, pdf);
	//wi.offset();

	if(pdf > 0.0f && !zero(Li)) {
		const float3 f = bsdf.f(hp.wo, wi.dir);
		if(!zero(f) && light.visible(geometry, wi))
			return f * Li * fmaxf(0.0f, dot(wi.dir, bsdf.N));
	}
	return make_float3(0.0f);
}

inline __device__ float3 estimateDirectRadiance(RNG* rng, const Geometry& geometry,
	const Light& light, const Shader& shader, const BSDF& bsdf, const HitPoint& hp)
{
	Ray wi;
	float3 Li, f;
	float lightPdf, bsdfPdf, weight;

	float3 L = make_float3(0.0f);
	for(unsigned int i=0; i<light.samples; i++) {
		float3 Ls = make_float3(0.0f);

		// Sample light
		wi = Ray(hp.position);
		Li = light.sampleL(rng, wi, lightPdf);
		//wi.offset();

		if(lightPdf > 0.0f && !zero(Li)) {
			float3 f = bsdf.f(hp.wo, wi.dir);
			if(!zero(f) && light.visible(geometry, wi)) {
				bsdfPdf = bsdf.pdf(hp.wo, wi.dir);
				weight  = powerHeuristic(1, lightPdf, 1, bsdfPdf);
				Ls = Ls + f * Li * (fmaxf(0.0f, dot(wi.dir, bsdf.N)) * weight / lightPdf);
			}
		}

		// Sample BSDF
		wi = Ray(hp.position);
		f  = bsdf.samplef(rng, hp.wo, wi.dir, bsdfPdf);
		//wi.offset();

		if(bsdfPdf > 0.0f && !zero(f)) {
			lightPdf = light.pdf(wi);
			weight   = powerHeuristic(1, bsdfPdf, 1, lightPdf);

			wi.t = Infinity;
			if(light.visible(geometry, wi)) {
				Li = light.L(-wi.dir);
				Ls = Ls + f * Li * (fmaxf(0.0f, dot(wi.dir, bsdf.N)) * weight / bsdfPdf);
			}
		}

		L = L + Ls;
	}

	return make_float3(
		L.x / light.samples,
		L.y / light.samples,
		L.z / light.samples);
}
