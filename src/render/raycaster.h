/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <core/raytracer.h>
#include <util/geometry.h>
#include <util/ray.h>

namespace Aurora {

class Raycaster : public Raytracer
{
public:
	Raycaster();
	~Raycaster();

	MStatus createFrame(const unsigned int width, const unsigned int height, Scene* scene, MDagPath& camera);
	MStatus destroyFrame();
	MStatus setRegion(const Rect& region);

	MStatus   render(bool ipr);
	RV_PIXEL* framebuffer();
	
protected:
	float4*	   m_pixels;
	Geometry   m_geometry;
	Ray*       m_rays;

	RV_PIXEL*  m_framebuffer;
	Rect       m_region;
	Dim		   m_size;
};

} // Aurora