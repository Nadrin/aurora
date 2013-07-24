/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <core/scene.h>

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

	MStatus	render(unsigned int width, unsigned int height, MDagPath& camera);

protected:
	Engine();

	Scene m_scene;
};

} // Aurora