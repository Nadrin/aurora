/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <util/math.h>

namespace Aurora {

class Ray 
{
public:
	gpu::float3 origin;
	gpu::float3 direction;
};

} // Aurora