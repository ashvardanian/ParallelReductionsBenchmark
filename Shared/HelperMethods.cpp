// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#include "HelperMethods.hpp"

using namespace nAV;

void nAV::gPrint(sBenchDescription const & p) {
    printf("Problem:\n");
    printf("- %s\n", p.mName.c_str());
    printf("- #%zu elements | #%zu elements per thread | #%zu threads\n",
           p.mSize, p.mSizeBatch, p.mCountThreads);
}

void nAV::gPrint(sBenchResult const * lArr, bSize const lLen,
                 sBenchResult const & lBaseline) {
    printf("\n");
    printf("- %s: sum = %.0f, dt = %.2f millisecs\n",
           lBaseline.mMethod.c_str(), lBaseline.mCheckSum, lBaseline.mDurationCompute);
    for (bSize i = 0; i < lLen; i ++) {
        sBenchResult const & r = lArr[i];
        printf("- %s: sum = %.0f, dt = %.2f millisecs,\t\timprovement = %.2fx\n",
               r.mMethod.c_str(), r.mCheckSum, r.mDurationCompute, lBaseline.mDurationCompute / r.mDurationCompute);
    }
    printf("\n");
}

void nAV::gPrint(sReal const * lArr, bSize const n) {
    printf("[");
    for (bSize j = 0; j < n; j ++)
        printf("%f,", lArr[j]);
    printf("]");
}

sStr nAV::gToCSV(SArr<sBenchResult> const & lArr) {
    std::stringstream lStream;
    {
        lStream << "Technology" << "," << "Device" << ",";
        lStream << "Task.Name" << "," << "Method" << ",";
        lStream << "Task.Size" << "," << "Task.Threads" << "," << "Task.BatchSize" << ",";
        lStream << "Duration.Compute" << "," << "Duration.Total" << "\n";
    }
    for (auto const & lResult : lArr) {
        lStream << lResult.mTechnology << "," << lResult.mDevice << ",";
        lStream << lResult.mTask.mName << "," << lResult.mMethod << ",";
        lStream << lResult.mTask.mSize << "," << lResult.mTask.mCountThreads << "," << lResult.mTask.mSizeBatch << ",";
        lStream << lResult.mDurationCompute << "," << lResult.mDurationTotal << "\n";
    }
    return lStream.str();
}

void nAV::gSaveToFile(sStr lPath, sStr lStr) {
    std::ofstream lStream(lPath.c_str());
    lStream << lStr;
    lStream.close();
}

sStr nAV::gReadFromFile(sStr lPath) {
    std::ifstream lStream(lPath.c_str());
    std::string lResult((std::istreambuf_iterator<char>(lStream)),
                        std::istreambuf_iterator<char>());
    return lResult;
}

void nAV::gThrow(sCStrC str) {
    throw std::runtime_error { str };
}

sReal nAV::gRandSmall() {
    return rand() % 25;
}
