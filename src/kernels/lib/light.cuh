/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

inline __device__ float3 Light::getL(const float3& P) const
{
	switch(type) {
	case DirectionalLight:
		return normalize(-direction);
	case PointLight:
	case AreaLight:
		return normalize(position - P);
	}
}