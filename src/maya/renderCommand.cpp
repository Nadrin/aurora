/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <maya/renderCommand.h>
#include <core/engine.h>

using namespace Aurora;

const char* RenderCommand::name = "auroraRender";

// Command parameters
namespace {
	bool			paramIprMode  = false;
	bool			paramIprPause = false;
	unsigned int	paramWidth	  = 512;
	unsigned int	paramHeight   = 512;
	MString			paramCamera   = "persp";
};

RenderCommand::RenderCommand()
{ }

RenderCommand::~RenderCommand()
{ }

void* RenderCommand::creator()
{
	return new RenderCommand();
}

MSyntax RenderCommand::newSyntax()
{
	MSyntax syntax;

	syntax.enableEdit(true);
	syntax.addFlag("-u", "-update");

	syntax.addFlag("-ipr", "-iprMode", MSyntax::kBoolean);
	syntax.addFlag("-p", "-pause", MSyntax::kBoolean);
	syntax.addFlag("-w", "-width", MSyntax::kLong);
	syntax.addFlag("-h", "-height", MSyntax::kLong);
	syntax.addFlag("-c", "-camera", MSyntax::kString);
	
	return syntax;
}

MStatus RenderCommand::parseSyntax(MArgDatabase& argdb)
{
	if(argdb.isFlagSet("-ipr"))
		argdb.getFlagArgument("-ipr", 0, paramIprMode);
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

	if(argdb.isEdit()) {
		if(argdb.isFlagSet("-u"))
			return Engine::instance()->update();

		if(argdb.isFlagSet("-ipr")) {
			if(paramIprMode)
				return Engine::instance()->iprRefresh();
			else
				return Engine::instance()->iprStop();
		}
		if(argdb.isFlagSet("-p"))
			return Engine::instance()->iprPause(paramIprPause);
	}
	else {
		if(argdb.isFlagSet("-ipr"))
			return Engine::instance()->iprStart(paramWidth, paramHeight, paramCamera);
		else
			return Engine::instance()->render(paramWidth, paramHeight, paramCamera);
	}
	return MS::kInvalidParameter;
}

