/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <render/photonMapper.h>
#include <kernels/kernels.h>

using namespace Aurora;

PhotonMapper::PhotonMapper() : m_pixels(NULL), m_rays(NULL), m_framebuffer(NULL), m_rng(NULL),
	m_primaryHits(NULL), m_lights(NULL) 
{
	m_numLights  = 0;
	m_numPhotons = 500;
}

PhotonMapper::~PhotonMapper()
{ }

MStatus PhotonMapper::createFrame(const unsigned int width, const unsigned int height,
	const unsigned short samples, Scene* scene, MDagPath& camera)
{
	const unsigned int numPixels = width * height;
	const unsigned int numRays   = width * height * samples;

	try {
		if(gpu::cudaMalloc(&m_rays, sizeof(Ray) * numRays) != gpu::cudaSuccess)
			throw std::exception();
		if(gpu::cudaMalloc(&m_primaryHits, sizeof(HitPoint) * numRays) != gpu::cudaSuccess)
			throw std::exception();
		if(gpu::cudaMalloc(&m_pixels, sizeof(float4) * numPixels) != gpu::cudaSuccess)
			throw std::exception();

		if((m_framebuffer = new RV_PIXEL[numPixels]) == NULL)
			throw std::exception();
	} 
	catch(const std::exception&) {
		destroyFrame();
		return MS::kInsufficientMemory;
	}

	m_scene    = scene;
	m_region   = Rect(0, width-1, 0, height-1);
	m_size     = Dim(width, height, samples);

	Renderer::generateRays(camera, m_size, m_region, m_rays, m_primaryHits);
	Renderer::setupRNG(&m_rng, m_numPhotons, 666);
	return MS::kSuccess;
}

MStatus PhotonMapper::update()
{
	if(m_lights)
		gpu::cudaFree(m_lights);

	m_numLights = cudaCreatePolyLights(m_scene->geometry(), m_scene->shaders(), &m_lights);

	if(!m_lights)
		return MS::kInsufficientMemory;
	return MS::kSuccess;
}

MStatus PhotonMapper::destroyFrame()
{
	if(m_rays) gpu::cudaFree(m_rays);
	if(m_primaryHits) gpu::cudaFree(m_primaryHits);
	if(m_pixels) gpu::cudaFree(m_pixels);
	if(m_rng) gpu::cudaFree(m_rng);
	if(m_lights) gpu::cudaFree(m_lights);

	delete[] m_framebuffer;

	m_rays        = NULL;
	m_primaryHits = NULL;
	m_pixels      = NULL;
	m_framebuffer = NULL;
	m_rng         = NULL;
	m_lights      = NULL;

	return MS::kSuccess;
}

MStatus PhotonMapper::setRegion(const Rect& region)
{
	m_region = region;
	return MS::kSuccess;
}

MStatus PhotonMapper::render(bool ipr)
{
	const unsigned int numRays = m_size.width * m_size.height * m_size.depth;
	gpu::cudaMemset(m_pixels, 0, sizeof(float4) * m_size.width * m_size.height);
	//Renderer::drawPixels(m_size, m_region, m_primaryHits, m_pixels);
	return MS::kSuccess;
}

RV_PIXEL* PhotonMapper::framebuffer()
{
	if(gpu::cudaMemcpy(m_framebuffer, m_pixels, sizeof(float4) * m_size.width * m_size.height, gpu::cudaMemcpyDeviceToHost) != gpu::cudaSuccess)
		return NULL;
	return m_framebuffer;
}