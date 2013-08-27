/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <render/photonMapper.h>
#include <kernels/kernels.h>

using namespace Aurora;

PhotonMapper::PhotonMapper() : m_pixels(NULL), m_rays(NULL), m_framebuffer(NULL), m_rng(NULL),
	m_primaryHits(NULL), m_emitters(NULL) 
{
	m_params.numEmitters        = 0;
	m_params.numLights          = 0;
	m_params.numHitPoints       = 0;

	m_params.numPhotons         = 500;
	m_params.numEmitterSamples  = 16;
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
		if(gpu::cudaMalloc(&m_photons, sizeof(Photon) * m_params.numPhotons) != gpu::cudaSuccess)
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
	Renderer::setupRNG(&m_rng, width * height * samples, GetTickCount());
	return MS::kSuccess;
}

MStatus PhotonMapper::update()
{
	if(m_emitters)
		gpu::cudaFree(m_emitters);

	m_params.numEmitters = cudaCreateEmitters(m_scene->geometry(), m_scene->shaders(), &m_emitters);
	if(!m_emitters) return MS::kInsufficientMemory;

	m_params.numHitPoints = m_size.width * m_size.height * m_size.depth;
	m_params.numLights    = (unsigned int)m_scene->lights().size;

	cudaRaycastPrimary(m_params, m_scene->geometry(), m_rays, m_primaryHits);
	return MS::kSuccess;
}

MStatus PhotonMapper::destroyFrame()
{
	gpu::cudaFree(m_rays);
	gpu::cudaFree(m_primaryHits);
	gpu::cudaFree(m_pixels);
	gpu::cudaFree(m_rng);
	gpu::cudaFree(m_emitters);
	gpu::cudaFree(m_photons);

	delete[] m_framebuffer;

	m_rays        = NULL;
	m_primaryHits = NULL;
	m_pixels      = NULL;
	m_framebuffer = NULL;
	m_rng         = NULL;
	m_emitters    = NULL;
	m_photons     = NULL;

	return MS::kSuccess;
}

MStatus PhotonMapper::setRegion(const Rect& region)
{
	m_region = region;
	return MS::kSuccess;
}

MStatus PhotonMapper::render(bool ipr)
{
	cudaPhotonTrace(m_params, m_rng, m_scene->geometry(), m_scene->shaders(), m_scene->lights(),
		m_emitters, m_photons, m_primaryHits);
	Renderer::drawPixels(m_size, m_region, m_primaryHits, m_pixels);
	return MS::kSuccess;
}

RV_PIXEL* PhotonMapper::framebuffer()
{
	if(gpu::cudaMemcpy(m_framebuffer, m_pixels, sizeof(float4) * m_size.width * m_size.height, gpu::cudaMemcpyDeviceToHost) != gpu::cudaSuccess)
		return NULL;
	return m_framebuffer;
}