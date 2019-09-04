// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#pragma once
#include "ComputeGraph.hpp"
#include "Shared/HelperAliasis.hpp"
#include "Shared/HelperMethods.hpp"
#define CL_SILENCE_DEPRECATION 1
#if __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace nCL {

    struct sTask;
    struct sQueue;
    struct sDevice;
    struct sVarWBuffer;

    struct sFileCompiled :
    public std::enable_shared_from_this<sFileCompiled> {
        ~sFileCompiled();
        
        SShared<sTask> fNewTask(sStr lKernelName);
        SWeak<nFlow::sFuncDectriptor> mSource;
        SShared<sDevice> mDevice;
        cl_program mProgram = nullptr;
    };
    
    struct sOperation {
        SShared<sQueue> mQueue;
        cl_event mHandle = nullptr;
        SArr<cl_event> mDependencies;
        
        bool fWaitUntilFinish();
    };
    
    struct sDataExchange :
    public std::enable_shared_from_this<sDataExchange> {
        nDisk::eMemoryAccess mType = nDisk::eMemoryAccess::kNone;
        sOperation mOperation;
        nFlow::sTaskDependancies mDependencies;
        cl_mem mOnGPU = nullptr;
        bPtrUntyped mOnCPU = nullptr;
        bSize mLengthBytes = 0;
    };
    
    struct sTask :
    public std::enable_shared_from_this<sTask> {
        ~sTask();
        cl_context fGetContext();
        
        SShared<sFileCompiled> mExe;
        sStr mKernelName;
        
        cl_kernel mKernel;
        SArr<SShared<sVarWBuffer>> mVarPtrs;
        SArr<bSize> mWorkDimensions;
        SArr<bSize> mGroupDimensions;
        SArr<bSize> mWorkOffsets;
        sOperation mOperation;
        
        bBool fInitKernelIfMissing();
        SShared<sVarWBuffer> fReallocVar(bSize);
        void fSetVar(bSize, SShared<sVarWBuffer>);
        void fRebind(bSize);
    };
    
    struct sVarWBuffer :
    public std::enable_shared_from_this<sVarWBuffer> {
        ~sVarWBuffer();
        cl_context fGetContext();
        
        SArr<SShared<sTask>> mUsedInTasks;
        SShared<nFlow::sArgDescriptor> mDescriptor;
        cl_mem mCLMemoryBuffer = nullptr;
        bPtrUntypedC mScalarValueAddress = nullptr;
        bSize mLength = 0;
        
        bBool fResetValue(bPtrUntypedC, bSize);
        bBool fResetConstant(bPtrUntypedC, bSize);
        bBool fResetVariable(bPtrUntyped, bSize);
        bBool fResetIndependantBuffer(bSize);
        bBool fPullInto(bPtrUntyped, bSize);
    private:
        bBool fResetAny(bSize, cl_mem);
    };
    
    struct sQueue :
    public std::enable_shared_from_this<sQueue> {
        ~sQueue();
        
        cl_command_queue mQueue = nullptr;
        
        void fWaitForCompletion();
        void fAppend(SShared<sTask> const &);
        void fAppend(SShared<sDataExchange> const &);
    };
    
    struct sDevice :
    public std::enable_shared_from_this<sDevice> {
        sDevice(cl_device_id);
        ~sDevice();
        
        SShared<sFileCompiled> fCompile(SShared<nFlow::sFuncDectriptor>);
        void fRunOnAnyQueue(SShared<sTask> const &);
        
        SArr<SShared<sQueue>> mQueues;
        std::map<sStr, SShared<sFileCompiled>> mExecutables;
        cl_device_id mDevice = nullptr;
        cl_context mContext = nullptr;
    private:
        void fInitContext();
        void fInitNewQueue();
    };
    
    struct sOpenCL {
        ~sOpenCL();
        
        SArr<cl_device_id> fAddAllDevices();
        SShared<sDevice> fFindDevice(cl_device_id);
        
        static void gPrintSpecs(cl_device_id);
        static void gCheckCode(cl_int err, sCStrC msg = nullptr);
        static SArr<cl_device_id> gDevicesID();
        static cl_program gCompile(cl_context, cl_device_id, sStr const & lContent);
        
        static cl_uint gGetSpecUInt(cl_device_id, cl_uint spec);
        static cl_ulong gGetSpecULong(cl_device_id, cl_uint spec);
        static sStr gGetSpecStr(cl_device_id, cl_uint spec);
        static sStr gGetName(cl_device_id);
    private:
        std::map<cl_device_id, SShared<sDevice>> mDevices;
    };

}
