/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <core/renderer.h>

#include <util/emitter.h>
#include <util/photon.h>

namespace Aurora {

class PhotonMapper : public Renderer
{
public:
	PhotonMapper();
	~PhotonMapper();

	MStatus createFrame(const unsigned int width, const unsigned int height, 
		const unsigned short samples, Scene* scene, MDagPath& camera);
	MStatus destroyFrame();
	MStatus setRegion(const Rect& region);

	MStatus	  update();

	MStatus   render(bool ipr);
	RV_PIXEL* framebuffer();
	
protected:
	float4*	   m_pixels;
	Scene*	   m_scene;
	Ray*       m_rays;
	RNG*       m_rng;

	HitPoint*    m_primaryHits;
	Emitter*     m_emitters;
	Photon*      m_photons;

	unsigned int m_numEmitters;
	unsigned int m_numPhotons;

	RV_PIXEL*  m_framebuffer;
	Rect       m_region;
	Dim        m_size;
};

} // Aurora