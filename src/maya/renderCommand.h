/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <maya/MPxCommand.h>
#include <maya/MSyntax.h>
#include <maya/MArgDatabase.h>

namespace Aurora {

class RenderCommand : public MPxCommand
{
public: // Member functions
	RenderCommand();
	~RenderCommand();

	virtual MStatus doIt(const MArgList& args);

	static MSyntax newSyntax();
	MStatus parseSyntax(MArgDatabase& argdb);

	static void* creator();

public: // Static variables
	static const char* name;
};

} // Aurora