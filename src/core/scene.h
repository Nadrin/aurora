/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <maya/MStatus.h>
#include <maya/MIntArray.h>
#include <maya/MDagPath.h>
#include <maya/MDagPathArray.h>
#include <maya/MFnDependencyNode.h>

#include <data/geometry.h>
#include <data/texture.h>

#include <util/array.h>
#include <util/shader.h>
#include <util/light.h>

namespace Aurora {

typedef std::map<std::string, unsigned int> IndexMap;

class Scene
{
protected:
	Geometry                    m_geometry;
	Array<Texture, HostMemory>  m_textures;
	Array<Shader, DeviceMemory> m_shaders;
	Array<Light, DeviceMemory>  m_lights;
public:
	Scene();
	~Scene();

	enum UpdateType {
		UpdateIpr,
		UpdateFull,
	};

	MStatus update(UpdateType type);
	Geometry& geometry() { return m_geometry; }

protected:
	MStatus updateMeshes(MDagPathArray &meshPaths, const IndexMap& shaderMap);
	MStatus updateShaders(MDagPathArray& shaderPaths, const IndexMap& textureMap, IndexMap& shaderMap);
	MStatus updateTextures(MDagPathArray& texturePaths, IndexMap& textureMap);

	static unsigned int getConnectedNode(const int type, MFnDependencyNode& depNode, const MString& attribute, const IndexMap& map);
	static unsigned int getNodeByName(const MString& name, const IndexMap& map);
	static void         getLocalIndices(const MIntArray& polygonIndices, const MIntArray& triangleIndices, MIntArray& localIndices);
};

} // Aurora