/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <maya/MGlobal.h>
#include <maya/MFnPlugin.h>

#include <maya/renderCommand.h>
#include <core/engine.h>

using namespace Aurora;

__declspec(dllexport)
MStatus initializePlugin(MObject obj)
{
	MStatus status;
	char initScriptPath[MAX_PATH];

	std::cerr << AURORA_BANNER << std::endl;

	MFnPlugin plugin(obj, AURORA_VENDOR, AURORA_VERSION, "4.5");
	plugin.registerCommand(RenderCommand::name, RenderCommand::creator, RenderCommand::newSyntax);
	
	GetFullPathName(AURORA_MELPATH, MAX_PATH, initScriptPath, NULL);
	if(!(status = MGlobal::sourceFile(initScriptPath))) {
		status.perror("Aurora");
		return status;
	}
	
	if(!(status = plugin.registerUI(AURORA_INITPROC, AURORA_DESTROYPROC))) {
		status.perror("Aurora");
		return status;
	}

	if(!(status = Engine::instance()->initialialize(0)))
		return status;
	return MS::kSuccess;
}

__declspec(dllexport)
MStatus uninitializePlugin(MObject obj)
{
	MFnPlugin plugin(obj);

	Engine::instance()->release();
	plugin.deregisterCommand(RenderCommand::name);
	return MS::kSuccess;
}