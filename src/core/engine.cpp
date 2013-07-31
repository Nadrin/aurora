/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <core/engine.h>
#include <render/debugPattern.h>
#include <render/raycaster.h>

#include <maya/MGlobal.h>
#include <maya/MRenderView.h>
#include <maya/MFnDagNode.h>
#include <maya/MItDag.h>

using namespace Aurora;

Engine::Engine() : m_deviceID(-1), m_scene(NULL), m_raytracer(NULL), m_state(Engine::StateIdle)
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

	MThreadAsync::init();

	m_deviceID  = deviceNumber;
	m_scene     = new Scene();
	m_raytracer = new Raycaster();

	return MS::kSuccess;
}

MStatus Engine::release()
{
	delete m_scene;
	delete m_raytracer;

	gpu::cudaDeviceReset();
	MThreadAsync::release();
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
	if(m_state != Engine::StateIdle)
		return MS::kFailure;
	if(!Engine::getRenderingCamera(camera, m_camera))
		return MS::kFailure;

	if(!m_raytracer->createFrame(width, height, m_scene, m_camera))
		return MS::kFailure;

	m_state  = Engine::StateIprRendering;
	m_window = Rect(0, width-1, 0, height-1);

	m_raytracer->setRegion(m_window);
	m_pause.unlock();

	MRenderView::setCurrentCamera(m_camera);
	MThreadAsync::createTask(Engine::renderThread, this, NULL, NULL);
	return MS::kSuccess;
}

MStatus Engine::iprPause(bool pause)
{
	if(pause) {
		m_state = Engine::StateIprPaused;
		m_pause.lock();
	}
	else {
		m_state = Engine::StateIprRendering;
		m_pause.unlock();
	}
	return MS::kSuccess;
}

MStatus Engine::iprRefresh()
{
	m_lock.lock();
	MStatus status = m_scene->update(Scene::NodeAll);
	m_lock.unlock();

	return status;
}

MStatus Engine::iprStop()
{
	m_state = Engine::StateIprStopped;
	m_pause.unlock();
	return MS::kSuccess;
}

MStatus Engine::render(unsigned int width, unsigned int height, const MString& camera)
{
	MStatus status;

	if(m_state != Engine::StateIdle)
		return MS::kFailure;
	if(!Engine::getRenderingCamera(camera, m_camera))
		return MS::kFailure;

	if((status = m_scene->update(Scene::NodeAll)) != MS::kSuccess) {
		m_state = Engine::StateIdle;
		return status;
	}

	if(!m_raytracer->createFrame(width, height, m_scene, m_camera))
		return MS::kFailure;

	m_window = Rect(0, width-1, 0, height-1);
	m_state  = Engine::StateRendering;

	m_raytracer->setRegion(m_window);
	MRenderView::setCurrentCamera(m_camera);

	if((status = m_raytracer->render(false)) != MS::kSuccess) {
		m_raytracer->destroyFrame();
		m_state = Engine::StateIdle;
		return status;
	}

	status = update(true);
	m_raytracer->destroyFrame();

	m_state = Engine::StateIdle;
	return status;
}

MStatus Engine::update(bool clearBackground)
{
	if(!m_lock.tryLock())
		return MS::kFailure;

	MRenderView::startRender(m_window.right+1, m_window.top+1, !clearBackground, true);
	MRenderView::updatePixels(m_window.left, m_window.right, m_window.bottom, m_window.top,
		m_raytracer->framebuffer(), true);
	MRenderView::endRender();

	if(m_state == Engine::StateIprUpdate)
		m_state = Engine::StateIprRendering;
	m_lock.unlock();
	return MS::kSuccess;
}

MThreadRetVal Engine::renderThread(void* context)
{
	Engine* engine = (Engine*)context;
	gpu::cudaSetDevice(engine->m_deviceID);

	while(engine->m_state != Engine::StateIprStopped) {
		engine->m_pause.lock();
		engine->m_pause.unlock();

		if(!engine->m_raytracer->render(true)) {
			Sleep(1); // Yield CPU time
			continue;
		}

		if(engine->m_state == Engine::StateIprStopped)
			break;

		if(engine->m_lock.tryLock()) {
			if(engine->m_state != Engine::StateIprUpdate) {
				engine->m_state = Engine::StateIprUpdate;
				MGlobal::executeCommandOnIdle("auroraRender -e -u;");
			}
			engine->m_lock.unlock();
		}
	}

	engine->m_raytracer->destroyFrame();
	engine->m_state = Engine::StateIdle;
	return 0;
}