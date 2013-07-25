/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <core/scene.h>
#include <core/raytracer.h>

#include <maya/MStatus.h>
#include <maya/MDagPath.h>

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
	MStatus update();

protected:
	Engine();
	static MStatus getRenderingCamera(const MString& name, MDagPath& path);

	int        m_deviceID;
	Scene*     m_scene;
	Raytracer* m_raytracer;
};

} // Aurora