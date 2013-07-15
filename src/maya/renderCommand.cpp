/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <maya/renderCommand.h>
#include <core/engine.h>

#include <maya/MRenderView.h>
#include <maya/MItDag.h>
#include <maya/MFnDagNode.h>

using namespace Aurora;

const char* RenderCommand::name = "auroraRender";

// Command parameters
namespace {
	bool			paramIprMode  = false;
	bool			paramIprPause = false;
	bool			paramEditMode = false;
	unsigned int	paramWidth	  = 512;
	unsigned int	paramHeight   = 512;
	MString			paramCamera   = "persp";
};

RenderCommand::RenderCommand()
{

}

RenderCommand::~RenderCommand()
{

}

void* RenderCommand::creator()
{
	return new RenderCommand();
}

MSyntax RenderCommand::newSyntax()
{
	MSyntax syntax;

	syntax.enableEdit(true);
	syntax.addFlag("-ipr", "-iprMode");

	syntax.addFlag("-p", "-pause", MSyntax::kBoolean);
	syntax.addFlag("-w", "-width", MSyntax::kLong);
	syntax.addFlag("-h", "-height", MSyntax::kLong);
	syntax.addFlag("-c", "-camera", MSyntax::kString);
	
	return syntax;
}

MStatus RenderCommand::parseSyntax(MArgDatabase& argdb)
{
	paramEditMode = argdb.isEdit();
	paramIprMode  = argdb.isFlagSet("-ipr");

	if(argdb.isFlagSet("-p"))
		argdb.getFlagArgument("-p", 0, paramIprPause);
	if(argdb.isFlagSet("-w"))
		argdb.getFlagArgument("-width", 0, paramWidth);
	if(argdb.isFlagSet("-h"))
		argdb.getFlagArgument("-height", 0, paramHeight);
	if(argdb.isFlagSet("-c"))
		argdb.getFlagArgument("-camera", 0, paramCamera);

	return MS::kSuccess;
}

MStatus RenderCommand::getCameraDagPath(const MString& name, MDagPath& path)
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

MStatus RenderCommand::doIt(const MArgList& args)
{
	MArgDatabase argdb(syntax(), args);
	if(!parseSyntax(argdb)) {
		std::cerr << "Aurora: Invalid syntax." << std::endl;
		return MS::kInvalidParameter;
	}

	if(!MRenderView::doesRenderEditorExist()) {
		std::cerr << "Aurora: Rendering is not supported in batch mode." << std::endl;
		return MS::kFailure;
	}

	MDagPath camera;
	if(getCameraDagPath(paramCamera, camera) != MStatus::kSuccess) {
		std::cerr << "Aurora: Unable to locate active camera node!" << std::endl;
		return MS::kFailure;
	}
	return Engine::instance()->render(paramWidth, paramHeight, camera);
}

