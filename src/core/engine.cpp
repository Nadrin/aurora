/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <core/engine.h>
#include <render/monteCarloRaytracer.h>
#include <render/photonMapper.h>

#include <maya/MGlobal.h>
#include <maya/MRenderView.h>
#include <maya/MFnDagNode.h>
#include <maya/MItDag.h>

#include <maya/MSceneMessage.h>
#include <maya/MProgressWindow.h>

using namespace Aurora;

Engine::Engine() : m_deviceID(-1), m_scene(NULL), m_renderer(NULL), m_state(Engine::StateIdle)
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
	gpu::cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync | cudaDeviceMapHost);

	MThreadAsync::init();

	m_deviceID  = deviceNumber;
	m_scene     = new Scene();
	m_renderer  = new MonteCarloRaytracer();

	gpu::cudaEventCreate(&m_eventUpdate[0]);
	gpu::cudaEventCreate(&m_eventUpdate[1]);
	gpu::cudaEventCreate(&m_eventRender[0]);
	gpu::cudaEventCreate(&m_eventRender[1]);

	m_sceneNewCallback  = MSceneMessage::addCallback(MSceneMessage::kBeforeNew, Engine::freeResources, this);
	m_sceneOpenCallback = MSceneMessage::addCallback(MSceneMessage::kBeforeOpen, Engine::freeResources, this);
	return MS::kSuccess;
}

MStatus Engine::release()
{
	MSceneMessage::removeCallback(m_sceneNewCallback);
	MSceneMessage::removeCallback(m_sceneOpenCallback);

	delete m_scene;
	delete m_renderer;

	gpu::cudaEventDestroy(m_eventUpdate[0]);
	gpu::cudaEventDestroy(m_eventUpdate[1]);
	gpu::cudaEventDestroy(m_eventRender[0]);
	gpu::cudaEventDestroy(m_eventRender[1]);

	gpu::cudaDeviceReset();
	MThreadAsync::release();
	return MS::kSuccess;
}

MStatus Engine::getRenderingCamera(const MString& name, MDagPath& path)
{
	MStatus status = MS::kFailure;

	MItDag dagIterator(MItDag::kBreadthFirst, MFn::kCamera);
	for(; !dagIterator.isDone(); dagIterator.next()) {
		dagIterator.getPath(path);

		if(MFnDagNode(path).name() == name) {
			path.pop();
			status = MS::kSuccess;
			break;
		}
		else {
			path.pop();
			if(MFnDagNode(path).name() == name) {
				status = MS::kSuccess;
				break;
			}
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

	if(!m_scene->update(Scene::UpdateFull))
		return MS::kFailure;

	if(!m_renderer->createFrame(width, height, 1, m_scene, m_camera)) {
		m_scene->free();
		return MS::kFailure;
	}

	m_state  = Engine::StateIprRendering;
	m_window = Rect(0, width-1, 0, height-1);

	m_renderer->setRegion(m_window);
	if(!m_renderer->update()) {
		m_scene->free();
		return MS::kFailure;
	}

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
	MStatus status;
	m_lock.lock();

	if(status = m_scene->update(Scene::UpdateFull))
		status = m_renderer->update();

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
	float updateTime, renderTime;

	if(m_state != Engine::StateIdle)
		return MS::kFailure;
	if(!Engine::getRenderingCamera(camera, m_camera))
		return MS::kFailure;

	std::cerr << "[Aurora] Rendering ..." << std::endl;

	MProgressWindow::reserve();
	MProgressWindow::setInterruptable(true);
	MProgressWindow::setTitle("Rendering progress");
	MProgressWindow::startProgress();

	MProgressWindow::setProgressStatus("Updating geometry ...");
	MProgressWindow::setProgressRange(0, 1);
	refreshUI();

	gpu::cudaEventRecord(m_eventUpdate[0]);
	if((status = m_scene->update(Scene::UpdateFull)) != MS::kSuccess) {
		m_state = Engine::StateIdle;
		return status;
	}
	gpu::cudaEventRecord(m_eventUpdate[1]);

	if(!m_renderer->createFrame(width, height, 4, m_scene, m_camera)) {
		std::cerr << "[Aurora] Failed to create rendering context: Out of memory." << std::endl;
		return MS::kFailure;
	}

	m_window = Rect(0, width-1, 0, height-1);
	m_state  = Engine::StateRendering;

	m_renderer->setRegion(m_window);
	MRenderView::setCurrentCamera(m_camera);
	MRenderView::startRender(m_window.right+1, m_window.top+1, true, false);

	MProgressWindow::setProgressStatus("Raytracing ...");
	refreshUI();

	gpu::cudaEventRecord(m_eventRender[0]);
	if((status = m_renderer->update()) != MS::kSuccess) {
		std::cerr << "[Aurora] Renderer update failed." << std::endl;
		m_renderer->destroyFrame();

		MRenderView::endRender();
		MProgressWindow::endProgress();

		m_state = Engine::StateIdle;
		return status;
	}
	if((status = m_renderer->render(this, false)) != MS::kSuccess) {
		std::cerr << "[Aurora] Rendering frame failed." << std::endl;
		m_renderer->destroyFrame();

		MRenderView::endRender();
		MProgressWindow::endProgress();

		m_state = Engine::StateIdle;
		return status;
	}
	gpu::cudaEventRecord(m_eventRender[1]);

	status = update(false);
	MRenderView::endRender();
	MProgressWindow::endProgress();

	m_renderer->destroyFrame();
	m_scene->free();

	m_state = Engine::StateIdle;

	gpu::cudaEventElapsedTime(&updateTime, m_eventUpdate[0], m_eventUpdate[1]);
	gpu::cudaEventElapsedTime(&renderTime, m_eventRender[0], m_eventRender[1]);
	std::cerr << "[Aurora] Update time: " << updateTime << " ms" << std::endl;
	std::cerr << "[Aurora] Render time: " << renderTime << " ms" << std::endl;
	return status;
}

MStatus Engine::refreshUI()
{
	return MGlobal::executeCommand("refresh -f;");
}

MStatus Engine::update(bool ipr)
{
	if(!m_lock.tryLock())
		return MS::kFailure;

	gpu::cudaDeviceSynchronize();

	if(ipr)
		MRenderView::startRender(m_window.right+1, m_window.top+1, false, true);

	MRenderView::updatePixels(m_window.left, m_window.right, m_window.bottom, m_window.top,
		m_renderer->framebuffer(), true);

	if(ipr)
		MRenderView::endRender();
	else
		MGlobal::executeCommand("refresh -f;");

	if(m_state == Engine::StateIprUpdate) {
		m_scene->update(Scene::UpdateIpr);
		m_state = Engine::StateIprRendering;
	}

	m_lock.unlock();
	return MS::kSuccess;
}

MThreadRetVal Engine::renderThread(void* context)
{
	Engine* engine = (Engine*)context;

	gpu::cudaSetDevice(engine->m_deviceID);

	SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_BEGIN);
	while(engine->m_state != Engine::StateIprStopped) {
		engine->m_pause.lock();
		engine->m_pause.unlock();

		Sleep(AURORA_CPUYIELD_TIME);
		if(!engine->m_renderer->render(engine, true))
			continue;

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
	SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_END);

	engine->m_renderer->destroyFrame();
	engine->m_scene->free();
	engine->m_state = Engine::StateIdle;
	return 0;
}

void Engine::freeResources(void* context)
{
	Engine* engine = (Engine*)context;

	engine->iprStop();
	engine->m_scene->free();
	engine->m_state = Engine::StateIdle;
}