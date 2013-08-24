/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

// Sampling: Uniform unit hemisphere
inline __device__ float3 sampleHemisphere(const float u1, const float u2)
{
	return make_float3(
		cosf(TwoPi * u2) * sqrtf(fmaxf(0.0f, 1.0f - u1*u1)),
		sinf(TwoPi * u2) * sqrtf(fmaxf(0.0f, 1.0f - u1*u1)),
		u1);
}

inline __device__ float sampleHemispherePdf()
{
	return 0.5f * InvPi;
}

// Sampling: Uniform unit disk
inline __device__ float2 sampleDisk(const float u1, const float u2)
{
	const float r     = sqrtf(u1);
	const float theta = TwoPi * u2;
	return make_float2(
		r * cosf(theta),
		r * sinf(theta));
}

// Sampling: Concentric unit disk
inline __device__ float2 sampleDiskConcentric(const float u1, const float u2)
{
	const float x = 2.0f * u1 - 1.0f;
	const float y = 2.0f * u2 - 1.0f;

	float r, theta;
	if(x >= -y) {
		if(x > y) { r=x; theta=Pi/4.0f * (y/x); }
		else      { r=y; theta=Pi/4.0f * (2.0f - x/y); }
	}
	else {
		if(x < y) { r=-x; theta=Pi/4.0f * (4.0f + y/x); }
		else { 
			r=-y;
			if(y != 0.0f) theta=Pi/4.0f * (6.0f - x/y);
			else          theta=0.0f;
		}
	}

	return make_float2(
		r * cosf(theta),
		r * sinf(theta));
}

// Sampling: Uniform unit hemisphere (cosine weighted)
inline __device__ float3 sampleHemisphereCosine(const float u1, const float u2)
{
	const float2 disk = sampleDiskConcentric(u1, u2);
	return make_float3(
		disk.x, disk.y,
		sqrtf(fmaxf(0.0f, 1.0f - disk.x*disk.x - disk.y*disk.y)));
}

inline __device__ float sampleHemisphereCosinePdf(const float costheta)
{
	return costheta * InvPi;
}

// Sampling: Uniform triangle
inline __device__ void sampleTriangle(const float u1, const float u2, float& u, float& v)
{
	const float sqrtu1 = sqrtf(u1);
	u = 1.0f - sqrtu1;
	v = u2 * sqrtu1;
}

// Sampling: 1D discrete poly light array
inline __device__ unsigned int sampleLightArray(const float u, const unsigned int numLights, const PolyLight* lights)
{
	int imin = 0;
	int imax = numLights-1;
	float distmin, distmax;

	do {
		const int imid = (imin + imax) / 2;
		distmin = fabsf(lights[imin].cdf - u);
		distmax = fabsf(lights[imax].cdf - u);

		if(distmin < distmax)
			imax  = imid;
		else 
			imin  = imid;
	} while((imax - imin) > 1);

	distmin = fabsf(lights[imin].cdf - u);
	distmax = fabsf(lights[imax].cdf - u);

	if(distmin < distmax) return imin;
	else return imax;
}

// Monte Carlo heuristics
inline __device__ float balanceHeuristic(
	const unsigned int n1, const float pdf1,
	const unsigned int n2, const float pdf2)
{
	return (n1 * pdf1) / (n1 * pdf1 + n2 * pdf2);
}

inline __device__ float powerHeuristic(
	const unsigned int n1, const float pdf1,
	const unsigned int n2, const float pdf2)
{
	const float f = n1 * pdf1;
	const float g = n2 * pdf2;
	return (f*f) / (f*f + g*g);
}
