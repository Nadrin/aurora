/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <maya/MStatus.h>
#include <maya/MDagPath.h>
#include <maya/MRenderView.h>

#include <util/math.h>
#include <core/scene.h>

namespace Aurora {

class Raytracer 
{
public:
	virtual MStatus createFrame(const unsigned int width, const unsigned int height, Scene* scene, MDagPath& camera) = 0;
	virtual MStatus destroyFrame() = 0;
	virtual MStatus setRegion(const Rect& region) = 0;

	virtual MStatus   render(bool ipr) = 0;
	virtual RV_PIXEL* framebuffer() = 0;
};

} // Aurora