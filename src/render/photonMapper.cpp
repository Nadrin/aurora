/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <render/photonMapper.h>
#include <kernels/kernels.h>

#include <maya/MProgressWindow.h>

using namespace Aurora;

PhotonMapper::PhotonMapper() : m_pixels(NULL), m_rays(NULL), m_framebuffer(NULL), m_rng(NULL),
	m_hits(NULL), m_emitters(NULL)
{
	m_params.numEmitters        = 0;
	m_params.numLights          = 0;
	m_params.numHitPoints       = 0;

	m_params.numPhotons         = 5000;
	m_params.numEmitterSamples  = 16;
	m_params.maxPhotonDepth     = 16;
	m_params.maxRayDepth        = 4;

	m_params.lightCDF = NULL;
}

PhotonMapper::~PhotonMapper()
{ }

MStatus PhotonMapper::createFrame(const unsigned int width, const unsigned int height,
	const unsigned short samples, Scene* scene, MDagPath& camera)
{
	const unsigned int numRays = width * height;

	try {
		if(gpu::cudaMalloc(&m_rays, sizeof(Ray) * numRays) != gpu::cudaSuccess)
			throw std::exception();
		if(gpu::cudaMalloc(&m_hits, sizeof(HitPoint) * numRays) != gpu::cudaSuccess)
			throw std::exception();
		if(gpu::cudaMalloc(&m_pixels, sizeof(float4) * numRays) != gpu::cudaSuccess)
			throw std::exception();
		if(gpu::cudaMalloc(&m_photons, sizeof(Photon) * m_params.numPhotons) != gpu::cudaSuccess)
			throw std::exception();

		if((m_framebuffer = new RV_PIXEL[numRays]) == NULL)
			throw std::exception();
	} 
	catch(const std::exception&) {
		destroyFrame();
		return MS::kInsufficientMemory;
	}

	m_camera   = camera;
	m_scene    = scene;
	m_region   = Rect(0, width-1, 0, height-1);
	m_size     = Dim(width, height, samples);

	Renderer::setupRNG(&m_rng, numRays, GetTickCount());
	return MS::kSuccess;
}

MStatus PhotonMapper::update()
{
	gpu::cudaFree(m_emitters);
	gpu::cudaFree(m_params.lightCDF);

//	m_params.numEmitters  = cudaCreateEmitters(m_scene->geometry(), m_scene->shaders(), &m_emitters);
	m_params.numLights    = cudaCreateLightsCDF(m_scene->geometry(), m_scene->lights(), &m_params.lightCDF);
	m_params.numHitPoints = m_size.width * m_size.height;
	return MS::kSuccess;
}

MStatus PhotonMapper::destroyFrame()
{
	gpu::cudaFree(m_params.lightCDF);
	gpu::cudaFree(m_emitters);

	m_params.lightCDF = NULL;

	gpu::cudaFree(m_rays);
	gpu::cudaFree(m_hits);
	gpu::cudaFree(m_pixels);
	gpu::cudaFree(m_rng);
	gpu::cudaFree(m_photons);

	delete[] m_framebuffer;

	m_rays        = NULL;
	m_hits        = NULL;
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
	const float weight = 1.0f / m_size.depth;

	MProgressWindow::reserve();
	MProgressWindow::setInterruptable(true);
	MProgressWindow::setTitle("Rendering ...");
	MProgressWindow::setProgressRange(0, m_size.depth);
	MProgressWindow::startProgress();

	Renderer::clearPixels(m_size, m_pixels);
	for(unsigned short i=0; i<m_size.depth; i++) {

		Renderer::generateRays(m_camera, m_size, m_region, i, m_rays, m_hits);
		cudaRaycastPrimary(m_params, m_scene->geometry(), m_rays, m_hits);
		cudaPhotonTrace(m_params, m_rng, m_scene->geometry(), m_scene->shaders(), m_scene->lights(),
			m_emitters, m_photons, m_hits);
		Renderer::drawPixels(m_size, m_region, m_hits, weight, m_pixels);

		if(MProgressWindow::isCancelled())
			break;
		MProgressWindow::advanceProgress(1);
		Sleep(0);
	}

	MProgressWindow::endProgress();
	return MS::kSuccess;
}

RV_PIXEL* PhotonMapper::framebuffer()
{
	if(gpu::cudaMemcpy(m_framebuffer, m_pixels, sizeof(float4) * m_size.width * m_size.height, gpu::cudaMemcpyDeviceToHost) != gpu::cudaSuccess)
		return NULL;
	return m_framebuffer;
}