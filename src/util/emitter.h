/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

namespace Aurora {

class Emitter
{
public:
	//float area;
	float emission;
	float cdf;
	unsigned int triangleID;
};

} // Aurora