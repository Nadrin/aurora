/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

inline __device__ BSDF Shader::getBSDF(const Geometry& geometry, const unsigned int index, 
	const float u, const float v) const
{
	BSDF bsdf;
	getBasisVectors(geometry, index, u, v, bsdf.N, bsdf.S, bsdf.T);

	switch(type) {
	case Shader::LambertShader:
		bsdf.type   = BSDF::BSDF_Lambert;
		bsdf.color1 = diffuseColor * diffuse;
		break;
	case Shader::PhongShader:
		bsdf.type     = BSDF::BSDF_Phong;
		bsdf.color1   = diffuseColor * diffuse;
		bsdf.color2   = specularColor;
		bsdf.exponent = exponent;
		break;
	}
	return bsdf;
}