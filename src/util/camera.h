/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <util/ray.h>

namespace Aurora {

class Camera
{
public:
	gpu::float3 position;
	gpu::float3 direction;
	gpu::float2 fov;
};

} // Aurora