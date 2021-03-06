/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <core/renderer.h>
#include <render/photonMapperParams.h>

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
	MDagPath   m_camera;
	Ray*       m_rays;
	RNG*       m_rng;

	HitPoint*    m_hits;
	Emitter*     m_emitters;
	Photon*      m_photons;

	PhotonMapperParams m_params;

	RV_PIXEL*  m_framebuffer;
	Rect       m_region;
	Dim        m_size;
};

} // Aurora