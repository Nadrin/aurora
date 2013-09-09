/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <core/engine.h>
#include <render/monteCarloRaytracer.h>
#include <kernels/kernels.h>

#include <maya/MProgressWindow.h>

#define TILE_SIZE (256*256)

using namespace Aurora;

MonteCarloRaytracer::MonteCarloRaytracer() : m_pixels(NULL), m_rays(NULL), m_framebuffer(NULL), m_rng(NULL), m_hit(NULL)
{ }

MonteCarloRaytracer::~MonteCarloRaytracer()
{ }

MStatus MonteCarloRaytracer::createFrame(const unsigned int width, const unsigned int height,
	const unsigned short samples, Scene* scene, MDagPath& camera)
{
	const unsigned int numRays   = width * height;

	if(gpu::cudaMalloc(&m_rays, sizeof(Ray) * numRays) != gpu::cudaSuccess)
		return MS::kInsufficientMemory;
	if(gpu::cudaMalloc(&m_hit, sizeof(HitPoint) * numRays) != gpu::cudaSuccess) {
		gpu::cudaFree(m_rays);
		return MS::kInsufficientMemory;
	}
	if(gpu::cudaMalloc(&m_pixels, sizeof(float4) * numRays) != gpu::cudaSuccess) {
		gpu::cudaFree(m_rays);
		gpu::cudaFree(m_hit);
		return MS::kInsufficientMemory;
	}

	m_framebuffer = new RV_PIXEL[numRays];
	if(m_framebuffer == NULL) {
		gpu::cudaFree(m_rays);
		gpu::cudaFree(m_hit);
		gpu::cudaFree(m_pixels);
		return MS::kInsufficientMemory;
	}

	m_scene    = scene;
	m_camera   = camera;
	m_region   = Rect(0, width-1, 0, height-1);
	m_size     = Dim(width, height, samples);

	//Renderer::generateRays(camera, m_size, m_region, m_rays, m_hit);
	Renderer::setupRNG(&m_rng, numRays, GetTickCount());
	return MS::kSuccess;
}

MStatus MonteCarloRaytracer::destroyFrame()
{
	gpu::cudaFree(m_rays);
	gpu::cudaFree(m_hit);
	gpu::cudaFree(m_pixels);
	gpu::cudaFree(m_rng);

	delete[] m_framebuffer;

	m_rays        = NULL;
	m_hit         = NULL;
	m_pixels      = NULL;
	m_framebuffer = NULL;
	m_rng         = NULL;
	return MS::kSuccess;
}

MStatus MonteCarloRaytracer::setRegion(const Rect& region)
{
	m_region = region;
	return MS::kSuccess;
}

MStatus MonteCarloRaytracer::render(Engine* engine, bool ipr)
{
	const float weight = 1.0f / m_size.depth;
	const unsigned int numRays  = m_size.width * m_size.height;
	const unsigned int numTiles = (numRays / TILE_SIZE) + ((numRays % TILE_SIZE) > 0);

	Renderer::clearPixels(m_size, m_pixels);

	if(!ipr) {
		MProgressWindow::setProgressRange(0, m_size.depth * numTiles);
		engine->refreshUI();
	}

	for(unsigned short i=0; i<m_size.depth; i++) {
		Renderer::generateRays(m_camera, m_size, m_region, i, m_rays, m_hit);
		
		for(unsigned int j=0; j<numTiles; j++) {
			const unsigned int offset      = j*TILE_SIZE;
			const unsigned int numTileRays = min(numRays - offset, unsigned int(TILE_SIZE));

			cudaRaytraceMonteCarlo(m_scene->geometry(),
				m_scene->shaders(), m_scene->textures(), m_scene->lights(), 
				m_rng, offset, numTileRays, m_rays, m_hit);

			if(!ipr) {
				gpu::cudaDeviceSynchronize();

				if(MProgressWindow::isCancelled())
					return MS::kSuccess;
				MProgressWindow::advanceProgress(1);
				engine->refreshUI();
			}
		}

		Renderer::drawPixels(m_size, m_region, m_hit, weight, m_pixels);
	}

	//Renderer::filterPixels(m_size, m_region, (void**)&m_pixels);
	return MS::kSuccess;
}

RV_PIXEL* MonteCarloRaytracer::framebuffer()
{
	if(gpu::cudaMemcpy(m_framebuffer, m_pixels, sizeof(float4) * m_size.width * m_size.height, gpu::cudaMemcpyDeviceToHost) != gpu::cudaSuccess)
		return NULL;
	return m_framebuffer;
}