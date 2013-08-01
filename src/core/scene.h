/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <maya/MStatus.h>
#include <maya/MDagPath.h>
#include <maya/MDagPathArray.h>

#include <core/geometry.h>

namespace Aurora {

class Scene
{
protected:
	Geometry m_geometry;
public:
	Scene();
	~Scene();

	enum UpdateType {
		NodeNone     = 0x00,
		NodeGeometry = 0x01,
		NodeAll      = 0xFF,
	};

	MStatus update(const int updateMask);
	Geometry& geometry() { return m_geometry; }

protected:
	MStatus updateMeshes(MDagPathArray &meshes, MStatus &status);
};

} // Aurora