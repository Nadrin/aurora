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
		BlinnShader,
	};

	ShaderType type;

	float3     color;
	float      diffuse;
	float	   emission;

	__device__ BSDF getBSDF(const Geometry& geometry, const unsigned int index,
		const float u, const float v) const;
};

typedef Array<Shader, DeviceMemory> ShadersArray;

} // Aurora