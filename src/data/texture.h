/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

namespace Aurora {

class Texture
{
public:
	__host__ bool load();
	__host__ void free();
};

} // Aurora