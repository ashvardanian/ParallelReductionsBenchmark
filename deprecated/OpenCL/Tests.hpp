// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#pragma once
#include "HelperAliasis.hpp"
#define CL_SILENCE_DEPRECATION 1
#if __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <math.h>
#include <stdio.h>
#include <vector>

namespace nCL {
        
    void gBenchReduce(sBenchDescription, SArr<sReal> const &, SArr<sBenchResult> &);
    void gBenchParallel(sBenchDescription, SArr<sReal> const &, SArr<sBenchResult> &);
    void gBenchParallel(sBenchDescription, SArr<sReal> const &, SArr<sBenchResult> &);
    
}
