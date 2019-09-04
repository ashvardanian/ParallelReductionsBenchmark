// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#pragma once
#include "HelperAliasis.hpp"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace nAV {
    
    template <typename aFunc>
    sMilliSec GMeasureTime(aFunc const & func) {
        struct timespec lStart, lStop;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &lStart);
        func();
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &lStop);
        return ((lStop.tv_sec - lStart.tv_sec) * 1e3 +
                (lStop.tv_nsec - lStart.tv_nsec) / 1e6);
    }
    
    template <typename aNum>
    void gResetOnCPU(SArr<aNum> & x) {
        std::fill(x.begin(), x.end(), 0);
    }
    
    void gPrint(sBenchDescription const &);
    void gPrint(sBenchResult const * lArr, bSize const lLen,
                sBenchResult const & lBaseline);
    void gPrint(sReal const * lArr, bSize const n);
    sStr gToCSV(SArr<sBenchResult> const & lArr);
    void gSaveToFile(sStr lPath, sStr lStr);
    sStr gReadFromFile(sStr lPath);
    
    void gThrow(sCStrC str);
    sReal gRandSmall();
    
}
