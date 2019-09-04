// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#pragma once
#include "HelperAliasis.hpp"

namespace nCPU {
    void gBenchReduce(sBenchDescription, SArr<sReal> const &, SArr<sBenchResult> &);
    void gBenchParallel(sBenchDescription, SArr<sReal> const &, SArr<sBenchResult> &);    
}
