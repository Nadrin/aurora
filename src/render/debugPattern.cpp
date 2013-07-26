/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <render/DebugPattern.h>

using namespace Aurora;

DebugPattern::DebugPattern() : m_framebuffer(NULL)
{ 
	m_size[0]   = m_size[1]   = 0;
	m_offset[0] = m_offset[1] = 0;
}

DebugPattern::~DebugPattern()
{ }

MStatus DebugPattern::createFrame(const unsigned int width, const unsigned int height, Scene* scene, MDagPath& camera)
{
	m_size[0]     = width;
	m_size[1]     = height;
	m_region      = Rect(0, width-1, 0, height-1);
	m_framebuffer = new RV_PIXEL[width*height];

	if(!m_framebuffer)
		return MS::kFailure;
	return MS::kSuccess;
}

MStatus DebugPattern::destroyFrame()
{
	delete[] m_framebuffer;
	m_framebuffer = NULL;
	return MS::kSuccess;
}

MStatus DebugPattern::setRegion(const Rect& region)
{
	m_region = region;
	return MS::kSuccess;
}

RV_PIXEL* DebugPattern::framebuffer()
{
	return m_framebuffer;
}

MStatus DebugPattern::render(bool ipr)
{
	for(unsigned int y=m_region.bottom; y<m_region.top; y++) {
		for(unsigned int x=m_region.left; x<m_region.right; x++) {
			unsigned int px = x + m_offset[0];
			unsigned int py = y + m_offset[1];

			RV_PIXEL* p = &m_framebuffer[y * m_size[0] + x];
			p->r = ((px ^ py) % 256) / 255.0f;
			p->g = ((px ^ py) % 256) / 255.0f;
			p->b = ((px ^ py) % 256) / 255.0f;
			p->a = 1.0f;
		}
	}

	m_offset[0]++;
	m_offset[1]++;
	return MS::kSuccess;
}