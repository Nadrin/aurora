/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <maya/MStatus.h>
#include <maya/MDagPath.h>
#include <maya/MDagPathArray.h>

#include <data/geometry.h>
#include <data/texture.h>

#include <util/shader.h>
#include <util/light.h>

namespace Aurora {

class Scene
{
protected:
	Geometry m_geometry;
	Texture* m_textures;
	Shader*  m_shaders;
	Light*   m_lights;
public:
	Scene();
	~Scene();

	enum UpdateType {
		NodeNone     = 0x00,
		NodeGeometry = 0x01,
		NodeShaders  = 0x02,
		NodeLights   = 0x04,
		NodeTextures = 0x08,
		NodeAll      = 0xFF,
	};

	MStatus update(const int updateMask);
	Geometry& geometry() { return m_geometry; }

protected:
	MStatus updateMeshes(MDagPathArray &meshes, MStatus &status);
};

} // Aurora