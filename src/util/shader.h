/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <data/geometry.h>

#include <util/array.h>
#include <util/bsdf.h>

namespace Aurora {

class Shader
{
public:
	enum ShaderType {
		LambertShader,
		PhongShader,
	};

	ShaderType type;

	float3	diffuseColor;
	float3	emissionColor;
	float3	specularColor;
	float3  reflectionColor;

	float	diffuse;
	float	emission;
	float	exponent;
	float	reflectivity;
	float	translucence;
	float	refractiveIndex;

	unsigned int texture;

	__device__ BSDF getBSDF(const Geometry& geometry, const unsigned int index,
		const float u, const float v) const;
};

typedef Array<Shader, DeviceMemory> ShadersArray;

} // Aurora