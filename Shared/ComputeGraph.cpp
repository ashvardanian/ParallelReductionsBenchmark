// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#include "ComputeGraph.hpp"
#include "HelperMethods.hpp"

using namespace nFlow;
using namespace nAV;

void sExeSource::fInitText() {
    if (!mContent.empty())
        return;
    mContent = gReadFromFile(mPath);
}

SShared<sFuncDectriptor> sExeSource::fMakeFunc(sStr xName, SArr<sArgDescriptor> xArgs) {
    auto lFunc = std::make_shared<sFuncDectriptor>();
    lFunc->mName = xName;
    lFunc->mSourceFile = shared_from_this();
    mFuncs.push_back(lFunc);
    for (auto & lArg : xArgs) {
        lFunc->mArgs.push_back(std::make_shared<sArgDescriptor>(lArg));
    }
    return lFunc;
}
