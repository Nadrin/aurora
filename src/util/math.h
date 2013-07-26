/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

namespace Aurora {

class Rect 
{
public:
	__host__ Rect() : left(0), right(0), top(0), bottom(0) { }
	__host__ Rect(unsigned int l, unsigned int r, unsigned int b, unsigned int t)
		: left(l), right(r), top(t), bottom(b) { }

	unsigned int left;
	unsigned int right;
	unsigned int bottom;
	unsigned int top;
};

} // Aurora