/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <render/raycaster.h>
#include <kernels/kernels.h>

#include <maya/MFnCamera.h>

using namespace Aurora;

Raycaster::Raycaster() : m_pixels(NULL), m_rays(NULL), m_framebuffer(NULL)
{ }

Raycaster::~Raycaster()
{ }

MStatus Raycaster::createFrame(const unsigned int width, const unsigned int height, Scene* scene, MDagPath& camera)
{
	const unsigned int numPixels = width * height;

	if(gpu::cudaMalloc(&m_rays, sizeof(Ray) * numPixels) != gpu::cudaSuccess)
		return MS::kInsufficientMemory;
	if(gpu::cudaMalloc(&m_pixels, sizeof(float4) * numPixels) != gpu::cudaSuccess) {
		gpu::cudaFree(m_rays);
		return MS::kInsufficientMemory;
	}

	m_framebuffer = new RV_PIXEL[numPixels];
	if(m_framebuffer == NULL) {
		gpu::cudaFree(m_rays);
		gpu::cudaFree(m_pixels);
		return MS::kInsufficientMemory;
	}

	m_geometry = scene->geometry();
	m_region   = Rect(0, width-1, 0, height-1);
	m_size     = Dim(width, height);

	Raytracer::generateRays(camera, m_size, m_region, m_rays);
	return MS::kSuccess;
}

MStatus Raycaster::destroyFrame()
{
	gpu::cudaFree(m_rays);
	gpu::cudaFree(m_pixels);
	delete[] m_framebuffer;

	m_rays        = NULL;
	m_pixels      = NULL;
	m_framebuffer = NULL;
	return MS::kSuccess;
}

MStatus Raycaster::setRegion(const Rect& region)
{
	m_region = region;
	return MS::kSuccess;
}

MStatus Raycaster::render(bool ipr)
{
	cudaRaycast(m_size.width * m_size.height, m_geometry, m_rays, m_pixels);
	return MS::kSuccess;
}

RV_PIXEL* Raycaster::framebuffer()
{
	if(gpu::cudaMemcpy(m_framebuffer, m_pixels, sizeof(float4) * m_size.width * m_size.height, gpu::cudaMemcpyDeviceToHost) != gpu::cudaSuccess)
		return NULL;
	return m_framebuffer;
}