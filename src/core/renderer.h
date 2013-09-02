/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <maya/MStatus.h>
#include <maya/MDagPath.h>
#include <maya/MRenderView.h>

#include <util/math.h>
#include <util/ray.h>
#include <util/hitpoint.h>
#include <core/scene.h>

namespace Aurora {

class Renderer
{
public:
	virtual MStatus createFrame(const unsigned int width, const unsigned int height,
		const unsigned short samples, Scene* scene, MDagPath& camera) = 0;
	virtual MStatus destroyFrame() = 0;
	virtual MStatus setRegion(const Rect& region) = 0;

	virtual MStatus	  update() { return MS::kSuccess; }
	
	virtual MStatus   render(bool ipr) = 0;
	virtual RV_PIXEL* framebuffer() = 0;

protected:
	static void generateRays(const MDagPath& camera, const Dim& size, const Rect& region,
		const unsigned short sampleID, Ray* rays, HitPoint* hit);
	static bool setupRNG(RNG** rng, const size_t count, const unsigned int seed);

	static void clearPixels(const Dim& size, void* pixels);
	static void drawPixels(const Dim& size, const Rect& region, const HitPoint* hit, const float weight, void* pixels);
	static void filterPixels(const Dim& size, const Rect& region, void** pixels);
};

} // Aurora