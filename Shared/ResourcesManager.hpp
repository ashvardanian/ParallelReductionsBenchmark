// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#pragma once
#include "HelperAliasis.hpp"
#include "ComputeGraph.hpp"
#define pFilePath(aFileName) "/Users/YourUsername/SandboxGPUs/" aFileName

namespace nAV {
        
    struct sResourcesManager {
        static sResourcesManager & gShared();
        SShared<nFlow::sExeSource> fLoad(sStr lPath);
    private:
        SDict<sStr, SShared<nFlow::sExeSource>> mSources;
    };
    
}
