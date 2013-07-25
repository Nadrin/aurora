/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <maya/MPxCommand.h>
#include <maya/MSyntax.h>
#include <maya/MArgDatabase.h>
#include <maya/MDagPath.h>

namespace Aurora {

class RenderCommand : public MPxCommand
{
public:
	RenderCommand();
	~RenderCommand();

	virtual MStatus doIt(const MArgList& args);

	static MSyntax newSyntax();
	MStatus parseSyntax(MArgDatabase& argdb);

	static void* creator();

public:
	static const char* name;
};

} // Aurora