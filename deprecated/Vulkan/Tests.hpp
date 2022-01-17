// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#pragma once
#include "HelperAliasis.hpp"

namespace nVK {
    void gBenchMandelbrot();
    void gBenchMandelbrotOld();
    void gBenchParallel(sBenchDescription, SArr<sReal> const &, SArr<sBenchResult> &);
}
