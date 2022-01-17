// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#pragma once
#include "Primitives.hpp"
#include <string>
#include <vector>
#include <map>

#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>

namespace nAV {
    
    using sReal = float;
    using sIdx = int;
    using sMilliSec = double;
    
    template <typename aObj>
    using SShared = std::shared_ptr<aObj>;
    template <typename aObj>
    using SWeak = std::weak_ptr<aObj>;
    template <typename aObj>
    using SArr = std::vector<aObj>;
    template <typename aKey, typename aObj>
    using SDict = std::map<aKey, aObj>;
    using sStr = std::string;
    template <typename aFirst, typename aSecond>
    using SPair = std::pair<aFirst, aSecond>;
    
    struct sBenchDescription {
        sStr mName;
        bSize mSize = 0;
        bSize mSizeBatch = 0;
        bSize mCountThreads = 0;
    };
    
    struct sBenchResult {
        sReal mCheckSum = 0;
        sMilliSec mDurationTotal = 0;
        sMilliSec mDurationCompute = 0;
        sStr mMethod;
        sStr mDevice;
        sStr mTechnology;
        sBenchDescription mTask;
    };
}

using namespace nAV;
