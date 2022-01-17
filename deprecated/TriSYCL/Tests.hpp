// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#pragma once
//#define TRISYCL_OPENCL 1
//#define BOOST_COMPUTE_USE_CPP11 1
#include "HelperAliasis.hpp"

namespace nSyCL {
    void gPrint();
    void gBenchParallel(sBenchDescription, SArr<sReal> const &, SArr<sBenchResult> &);
}
