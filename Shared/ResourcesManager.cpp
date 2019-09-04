// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#include "ResourcesManager.hpp"
#include "HelperMethods.hpp"

using namespace nFlow;
using namespace nAV;

sResourcesManager & sResourcesManager::gShared() {
    static sResourcesManager lShared;
    return lShared;
}

SShared<sExeSource> sResourcesManager::fLoad(sStr lPath) {
    auto lResult = std::make_shared<sExeSource>();
    lResult->mPath = lPath;
    mSources[lPath] = lResult;
    lResult->fInitText();
    return lResult;
}
