/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <util/math.h>
#include <core/scene.h>
#include <core/raytracer.h>

#include <maya/MStatus.h>
#include <maya/MDagPath.h>
#include <maya/MThreadAsync.h>
#include <maya/MMutexLock.h>

namespace Aurora {

class Engine 
{
public:
	static Engine* instance();
	~Engine();

	MStatus initialialize(const int device=0);
	MStatus release();

	MStatus iprStart(unsigned int width, unsigned int height, const MString& camera);
	MStatus iprPause(bool pause);
	MStatus iprRefresh();
	MStatus iprStop();

	MStatus	render(unsigned int width, unsigned int height, const MString& camera);
	MStatus update(bool clearBackground);

protected:
	Engine();

	static MStatus       getRenderingCamera(const MString& name, MDagPath& path);
	static MThreadRetVal renderThread(void* context);

	enum EngineState {
		StateIdle = 0,
		StateRendering,
		StateIprRendering,
		StateIprPaused,
		StateIprStopped,
		StateIprUpdate,
	};

	int         m_deviceID;
	Rect		m_window;
	Scene*      m_scene;
	Raytracer*  m_raytracer;
	EngineState m_state;

	MDagPath    m_camera;
	MMutexLock  m_lock;
	MMutexLock  m_pause;

	gpu::cudaEvent_t m_eventUpdate[2];
	gpu::cudaEvent_t m_eventRender[2];
};

} // Aurora