/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <core/raytracer.h>

namespace Aurora {

class DebugPattern : public Raytracer
{
public:
	DebugPattern();
	~DebugPattern();

	MStatus createFrame(const unsigned int width, const unsigned int height, Scene* scene, MDagPath& camera);
	MStatus destroyFrame();
	MStatus setRegion(const Rect& region);

	MStatus   render(bool ipr);
	RV_PIXEL* framebuffer();

protected:
	Dim			m_size;
	uint2		m_offset;
	Rect		m_region;
	RV_PIXEL*	m_framebuffer;
};

} // Aurora