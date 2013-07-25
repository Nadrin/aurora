/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <core/engine.h>

#include <maya/MRenderView.h>
#include <maya/MFnDagNode.h>
#include <maya/MItDag.h>

using namespace Aurora;

Engine::Engine() : m_deviceID(-1), m_scene(NULL), m_raytracer(NULL)
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

	m_deviceID  = deviceNumber;
	m_scene     = new Scene();
	m_raytracer = NULL;

	return MS::kSuccess;
}

MStatus Engine::release()
{
	delete m_scene;
	gpu::cudaDeviceReset();
	return MS::kSuccess;
}

MStatus Engine::getRenderingCamera(const MString& name, MDagPath& path)
{
	MStatus status = MS::kFailure;

	MItDag dagIterator(MItDag::kBreadthFirst, MFn::kCamera);
	for(; !dagIterator.isDone(); dagIterator.next()) {
		if(!dagIterator.getPath(path))
			continue;
		if(!path.pop())
			continue;

		MFnDagNode dagNode(path);
		if(dagNode.name() == name) {
			status = MS::kSuccess;
			break;
		}
	}
	return status;
}

MStatus Engine::iprStart(unsigned int width, unsigned int height, const MString& camera)
{
	std::cerr << "iprStart" << std::endl;
	return MS::kSuccess;
}

MStatus Engine::iprPause(bool pause)
{
	std::cerr << "iprPause: " << pause << std::endl;
	return MS::kSuccess;
}

MStatus Engine::iprRefresh()
{
	std::cerr << "iprRefresh" << std::endl;
	return MS::kSuccess;
}

MStatus Engine::iprStop()
{
	std::cerr << "iprStop" << std::endl;
	return MS::kSuccess;
}

MStatus	Engine::render(unsigned int width, unsigned int height, const MString& camera)
{
	MDagPath dagCamera;
	if(!Engine::getRenderingCamera(camera, dagCamera)) {
		std::cerr << "Aurora: Unable to locate active camera node!" << std::endl;
		return MS::kFailure;
	}

	MRenderView::setCurrentCamera(dagCamera);
	MRenderView::startRender(width, height, false, true);

#if 0
	if((status = m_scene->update(Scene::NodeAll)) != MS::kSuccess)
		return status;
#endif

	std::cerr << "render" << std::endl;

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

MStatus Engine::update()
{
	return MS::kSuccess;
}