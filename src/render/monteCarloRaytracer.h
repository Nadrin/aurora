/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <core/renderer.h>
#include <core/scene.h>
#include <util/ray.h>

namespace Aurora {

class MonteCarloRaytracer : public Renderer
{
public:
	MonteCarloRaytracer();
	~MonteCarloRaytracer();

	MStatus createFrame(const unsigned int width, const unsigned int height, 
		const unsigned short samples, Scene* scene, MDagPath& camera);
	MStatus destroyFrame();
	MStatus setRegion(const Rect& region);

	MStatus   render(Engine* engine, bool ipr);
	RV_PIXEL* framebuffer();
	
protected:
	float4*	   m_pixels;
	Scene*	   m_scene;
	MDagPath   m_camera;
	Ray*       m_rays;
	RNG*       m_rng;
	HitPoint*  m_hit;

	RV_PIXEL*  m_framebuffer;
	Rect       m_region;
	Dim        m_size;
};

} // Aurora