/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

namespace Aurora {

class HitPoint
{
public:
	__host__ __device__
	HitPoint()
		: triangleID(-1), u(0.0f), v(0.0f) { }

	float3 color;
	float3 position;

	float  u;
	float  v;
	int    triangleID;
};

} // Aurora