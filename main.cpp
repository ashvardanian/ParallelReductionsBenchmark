// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#include <iostream>
#include "Shared/Vulkan/Tests.hpp"
#include "Shared/TriSYCL/Tests.hpp"
#include "Shared/OpenCL/Tests.hpp"
#include "Shared/Halide/Tests.hpp"
#include "Shared/CPU/Tests.hpp"
#include "Shared/ArrayFire/Tests.hpp"
#include "Shared/HelperMethods.hpp"

void gEnque(SArr<sBenchDescription> & zArr,
            SArr<bSize> xProblemSizes, SArr<bSize> xGroupSizes, sStr xName) {
    for (bSize lTotal : xProblemSizes) {
        for (bSize lWorkItemSize : xGroupSizes) {
            zArr.push_back({ xName, lTotal, lWorkItemSize, lTotal / lWorkItemSize });
        }
    }
}


int main() {
    sStr const lFamilyDataParallel = "DataParallel";
    sStr const lFamilyReduction = "Reduction";
    sStr const lFamilyMatMul = "MatMul";

    SArr<sBenchDescription> lTasks;
    SArr<sBenchResult> lResultsAll;
    SArr<sReal> lArrFullCPU;
    {
        SArr<bSize> const lProblemSizesSmall = {
            1024*32, 1024*256, 1024*1024,
            // 1024*1024*32, 1024*1024*256
        };
        SArr<bSize> const lProblemSizesBig = {
            1024ULL*1024*512, 1024ULL*1024*1024*2
        };
        SArr<bSize> const lGroupSizesAll = {
            256, 128, 64, 32, 16, 8, 4,
        };
        SArr<bSize> const lGroupSizesMain = {
            128, 16, 1
        };

        gEnque(lTasks, lProblemSizesSmall, lGroupSizesAll, lFamilyReduction);
        gEnque(lTasks, lProblemSizesSmall, lGroupSizesMain, lFamilyDataParallel);

        lArrFullCPU.resize(lProblemSizesSmall.back());
        std::generate(lArrFullCPU.begin(), lArrFullCPU.end(), &nAV::gRandSmall);
    }
    
    std::for_each(lTasks.cbegin(), lTasks.cend(), [&](sBenchDescription const & lDesc) {
        if (lDesc.mName == lFamilyReduction) {
            if (lDesc.mSizeBatch == 1) {
                nCPU::gBenchReduce(lDesc, lArrFullCPU, lResultsAll);
            } else {
                nCL::gBenchReduce(lDesc, lArrFullCPU, lResultsAll);
            }
        } else {
            nCL::gBenchParallel(lDesc, lArrFullCPU, lResultsAll);
            nVK::gBenchParallel(lDesc, lArrFullCPU, lResultsAll);
            nHal::gBenchParallel(lDesc, lArrFullCPU, lResultsAll);
            if (lDesc.mSizeBatch == 1) {
                nAF::gBenchParallel(lDesc, lArrFullCPU, lResultsAll);
                nSyCL::gBenchParallel(lDesc, lArrFullCPU, lResultsAll);
                nCPU::gBenchParallel(lDesc, lArrFullCPU, lResultsAll);
            }
        }
    });
    auto const lStr = nAV::gToCSV(lResultsAll);
    nAV::gSaveToFile("Results.csv", lStr);    
    return 0;
}
