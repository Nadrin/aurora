/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <util/bsdf.h>

namespace Aurora {

class Shader
{
public:
	enum TextureChannel {
		ChannelColor = 0,
		ChannelCount,
	};

	BSDF bsdf;
	unsigned int texture[ChannelCount];

	float3 color;
	float3 ambientColor;
	float  diffuse;
};

} // Aurora