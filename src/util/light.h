/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <util/array.h>

namespace Aurora {

class Light
{
public:
	float  intensity;
	short  samples;
	float3 color;
};

typedef Array<Light, DeviceMemory>  LightsArray;

} // Aurora