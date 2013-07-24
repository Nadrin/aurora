/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <core/engine.h>

#include <maya/MRenderView.h>

using namespace Aurora;

Engine::Engine()
{ }

Engine::~Engine()
{ }

Engine* Engine::instance()
{
	static Engine engine;
	return &engine;
}

MStatus Engine::initialialize(const int device)
{
	int deviceCount;
	int deviceNumber = device;

	if(gpu::cudaGetDeviceCount(&deviceCount) != gpu::cudaSuccess) {
		std::cerr << "Aurora: No CUDA capable devices found. Exiting..." << std::endl;
		return MS::kFailure;
	}

	if(gpu::cudaSetDevice(deviceNumber) != gpu::cudaSuccess) {
		std::cerr << "Aurora: Failed to set CUDA device: " << deviceNumber << std::endl;
		return MS::kFailure;
	}
	gpu::cudaSetDeviceFlags(cudaDeviceScheduleAuto | cudaDeviceMapHost);
	return MS::kSuccess;
}

MStatus Engine::release()
{
	m_scene.geometry().free();
	gpu::cudaDeviceReset();
	return MS::kSuccess;
}

MStatus	Engine::render(unsigned int width, unsigned int height, MDagPath& camera)
{
	MStatus status;

	MRenderView::setCurrentCamera(camera);
	MRenderView::startRender(width, height, false, true);

	if((status = m_scene.update(Scene::NodeAll)) != MS::kSuccess)
		return status;

	// Test code!
	RV_PIXEL* pixels = new RV_PIXEL[width*height];
	for(unsigned int y=0; y<height; y++) {
		for(unsigned int x=0; x<width; x++) {
			RV_PIXEL p;
			int v = (x ^ y) % 256;
			p.r = v / 255.0f;
			p.g = v / 255.0f;
			p.b = v / 255.0f;
			p.a = 1.0f;

			pixels[y*width+x] = p;
		}
	}
	MRenderView::updatePixels(0, width-1, 0, height-1, pixels, true);
	delete[] pixels;

	MRenderView::endRender();
	return MS::kSuccess;
}