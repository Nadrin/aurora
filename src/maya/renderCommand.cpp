/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include "renderCommand.h"

using namespace Aurora;

const char* RenderCommand::name = "auroraRender";

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
	return syntax;
}

MStatus RenderCommand::parseSyntax(MArgDatabase& argdb)
{
	return MS::kSuccess;
}

MStatus RenderCommand::doIt(const MArgList& args)
{
	return MS::kSuccess;
}

