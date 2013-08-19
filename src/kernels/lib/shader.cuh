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
		bsdf.type = BSDF::BSDF_Lambert;
		bsdf.spectrum = color * diffuse;
		break;
	case Shader::PhongShader:
		bsdf.type = BSDF::BSDF_Phong;
		break;
	}
	return bsdf;
}