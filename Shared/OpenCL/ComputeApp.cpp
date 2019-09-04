// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#include "ComputeApp.hpp"
#include "HelperMethods.hpp"
#include <cmath>

using namespace nFlow;
using namespace nAV;
using namespace nCL;

void sDevice::fInitContext() {
    sDevice & lDevice = *this;
    cl_int lError = 0;
    lDevice.mContext = clCreateContext(NULL, 1, &lDevice.mDevice, NULL, NULL, &lError);
    sOpenCL::gCheckCode(lError, "Couldn't create a context");
}

void sDevice::fInitNewQueue() {
    sDevice & lDevice = *this;
    cl_int lError = 0;
    auto lQ = std::make_shared<sQueue>();
    lQ->mQueue = clCreateCommandQueue(lDevice.mContext, lDevice.mDevice, 0, &lError);
    sOpenCL::gCheckCode(lError, "Couldn't create a command queue");
    lDevice.mQueues.emplace_back(std::move(lQ));
}

SShared<sFileCompiled> sDevice::fCompile(SShared<sFuncDectriptor> xPtr) {
    sDevice & lDevice = *this;
    auto lResult = std::make_shared<sFileCompiled>();
    lResult->mProgram = sOpenCL::gCompile(lDevice.mContext, lDevice.mDevice, xPtr->mSourceFile->mContent.c_str());
    lResult->mDevice = lDevice.shared_from_this();
    lResult->mSource = xPtr;
    return lResult;
}

SShared<sTask> sFileCompiled::fNewTask(sStr xKernelName) {
    sFileCompiled & lExe = *this;
    auto lTaskPtr = std::make_shared<sTask>();
    lTaskPtr->mExe = lExe.shared_from_this();
    lTaskPtr->mKernelName = xKernelName;
    lTaskPtr->fInitKernelIfMissing();
    return lTaskPtr;
}

bBool sTask::fInitKernelIfMissing() {
    auto & lTask = *this;
    if (lTask.mKernel)
        return true;
    cl_int lError = 0;
    lTask.mKernel = clCreateKernel(lTask.mExe->mProgram, lTask.mKernelName.c_str(), &lError);
    sOpenCL::gCheckCode(lError, "Couldn't create a kernel");
    return true;
}

bBool sVarWBuffer::fPullInto(bPtrUntyped lData, bSize xLen) {
    auto lExchangePtr = std::make_shared<sDataExchange>();
    lExchangePtr->mLengthBytes = xLen;
    lExchangePtr->mOnCPU = lData;
    lExchangePtr->mType = nDisk::eMemoryAccess::kRead;
    lExchangePtr->mOnGPU = mCLMemoryBuffer;
    
    auto lQPtr = mUsedInTasks.front()->mOperation.mQueue;
    lQPtr->fAppend(lExchangePtr);
    lQPtr->fWaitForCompletion();
    return true;
}

bBool sVarWBuffer::fResetValue(bPtrUntypedC xData, bSize xLen) {
    mScalarValueAddress = xData;
    return fResetAny(xLen, NULL);
}

bBool sVarWBuffer::fResetConstant(bPtrUntypedC xData, bSize xLen) {
    cl_int lError;
    cl_mem lMem = clCreateBuffer(fGetContext(),
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 xLen, (bPtrUntyped)xData, &lError);
    sOpenCL::gCheckCode(lError, "Couldn't allocate buffer!");
    return fResetAny(xLen, lMem);
}

bBool sVarWBuffer::fResetVariable(bPtrUntyped xData, bSize xLen) {
    cl_int lError;
    cl_mem lMem = clCreateBuffer(fGetContext(),
                                 CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 xLen, xData, &lError);
    sOpenCL::gCheckCode(lError, "Couldn't allocate buffer!");
    return fResetAny(xLen, lMem);
}

bBool sVarWBuffer::fResetIndependantBuffer(bSize xLen) {
    return fResetAny(xLen, NULL);
}

bBool sVarWBuffer::fResetAny(bSize xLen, cl_mem lMem) {
    if (mCLMemoryBuffer)
        clReleaseMemObject(mCLMemoryBuffer);
    if (lMem)
        mScalarValueAddress = NULL;
    mLength = xLen;
    mCLMemoryBuffer = lMem;
    return true;
}

void sTask::fRebind(bSize xIdx) {
    auto & lTask = *this;
    auto lPtr = lTask.mVarPtrs[xIdx];
    cl_int lError;
    if (lPtr->mScalarValueAddress) {
        lError = clSetKernelArg(lTask.mKernel, static_cast<cl_uint>(xIdx),
                                lPtr->mLength, lPtr->mScalarValueAddress);
    } else if (lPtr->mCLMemoryBuffer) {
        lError = clSetKernelArg(lTask.mKernel, static_cast<cl_uint>(xIdx),
                                sizeof(lPtr->mCLMemoryBuffer), &lPtr->mCLMemoryBuffer);
    } else {
        lError = clSetKernelArg(lTask.mKernel, static_cast<cl_uint>(xIdx),
                                lPtr->mLength, NULL);
    }
    sOpenCL::gCheckCode(lError, "Couldn't create a kernel argument");
}

void sTask::fSetVar(bSize xIdx, SShared<sVarWBuffer> xPtr) {
    auto & lTask = *this;
    xPtr->mUsedInTasks.push_back(lTask.shared_from_this());
    if (lTask.mVarPtrs.size() <= xIdx)
        lTask.mVarPtrs.resize(xIdx + 1);
    lTask.mVarPtrs[xIdx] = xPtr;
}

SShared<sVarWBuffer> sTask::fReallocVar(bSize xIdx) {
    auto lPtr = std::make_shared<sVarWBuffer>();
    fSetVar(xIdx, lPtr);
    return lPtr;
}

cl_context sVarWBuffer::fGetContext() {
    return mUsedInTasks.front()->mExe->mDevice->mContext;
}

void sQueue::fAppend(SShared<sDataExchange> const & lTask) {
    auto & lQ = *this;
    lTask->mOperation.mQueue = lQ.shared_from_this();
    if (lTask->mType == nDisk::eMemoryAccess::kRead) {
        cl_int err = clEnqueueReadBuffer(lQ.mQueue, lTask->mOnGPU, CL_TRUE,
                                         0, lTask->mLengthBytes, lTask->mOnCPU,
                                         static_cast<cl_uint>(lTask->mOperation.mDependencies.size()),
                                         lTask->mOperation.mDependencies.data(),
                                         &lTask->mOperation.mHandle);
        sOpenCL::gCheckCode(err, "Couldn't read the buffer");
    } else {
        cl_int err = clEnqueueWriteBuffer(lQ.mQueue, lTask->mOnGPU, CL_TRUE,
                                          0, lTask->mLengthBytes, lTask->mOnCPU,
                                          static_cast<cl_uint>(lTask->mOperation.mDependencies.size()),
                                          lTask->mOperation.mDependencies.data(),
                                          &lTask->mOperation.mHandle);
        sOpenCL::gCheckCode(err, "Couldn't write the buffer");
    }
}

void sQueue::fAppend(SShared<sTask> const & lTaskPtr) {
    for (bSize lIdx = 0; lIdx < lTaskPtr->mVarPtrs.size(); lIdx ++)
        lTaskPtr->fRebind(lIdx);
    
    auto & lQ = *this;
    lTaskPtr->mOperation.mQueue = lQ.shared_from_this();
    auto lError = clEnqueueNDRangeKernel(lQ.mQueue,
                                         lTaskPtr->mKernel,
                                         static_cast<cl_uint>(lTaskPtr->mWorkDimensions.size()),
                                         lTaskPtr->mWorkOffsets.empty() ? NULL : lTaskPtr->mWorkOffsets.data(),
                                         lTaskPtr->mWorkDimensions.data(),
                                         lTaskPtr->mGroupDimensions.data(),
                                         static_cast<cl_uint>(lTaskPtr->mOperation.mDependencies.size()),
                                         lTaskPtr->mOperation.mDependencies.data(),
                                         &lTaskPtr->mOperation.mHandle);
    sOpenCL::gCheckCode(lError, "Couldn't enqueue the kernel");
}

void sQueue::fWaitForCompletion() {
    clFinish(mQueue);
}

void sDevice::fRunOnAnyQueue(SShared<sTask> const & lTaskPtr) {
    SShared<sQueue> const & lQPtr = mQueues.front();
    lQPtr->fAppend(lTaskPtr);
    lQPtr->fWaitForCompletion();
}

SArr<cl_device_id> sOpenCL::fAddAllDevices() {
    auto lDevIDs = gDevicesID();
    for (auto lDevID : lDevIDs)
        mDevices[lDevID] = std::make_shared<sDevice>(lDevID);
    return lDevIDs;
}

SShared<sDevice> sOpenCL::fFindDevice(cl_device_id lDevID) {
    return mDevices[lDevID];
}

sDevice::sDevice(cl_device_id lID) : mDevice(lID) {
    fInitContext();
    fInitNewQueue();
}

sOpenCL::~sOpenCL() { }
sDevice::~sDevice() { clReleaseContext(mContext); }
sQueue::~sQueue() { clReleaseCommandQueue(mQueue); }
sFileCompiled::~sFileCompiled() { clReleaseProgram(mProgram); }
sVarWBuffer::~sVarWBuffer() { clReleaseMemObject(mCLMemoryBuffer); }
sTask::~sTask() { }

cl_program sOpenCL::gCompile(cl_context lCtx, cl_device_id lDev, sStr const & lContent) {
    sCStrC lCodeTextPtr = lContent.c_str();
    bSize const lCodeLen = lContent.size();
    
    cl_int err;
    cl_program lProgram = clCreateProgramWithSource(lCtx, 1, (sCStrC *)&lCodeTextPtr, &lCodeLen, &err);
    gCheckCode(err, "Couldn't create the program");
    
    // The third parameter is the list of deives.
    // If it's NULL, the program executable is built for all devices
    // associated with program for which a source or binary has been loaded.
    //
    // The fourth parameter accepts options that configure the compilation.
    // These are similar to the flags used by gcc. For example, you can
    // define a macro with the option -DMACRO=VALUE and turn off optimization
    // with -cl-opt-disable.
    //
    // Docs: https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clBuildProgram.html
    err = clBuildProgram(lProgram, 1, &lDev, NULL, NULL, NULL);
    if (err < 0) {
        // Find size of log and print to std output.
        bSize lLogLen = 0;
        clGetProgramBuildInfo(lProgram, lDev, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &lLogLen);
        sStr lLog;
        lLog.resize(lLogLen);
        clGetProgramBuildInfo(lProgram, lDev, CL_PROGRAM_BUILD_LOG,
                              lLogLen + 1, (bPtrUntyped)lLog.c_str(), NULL);
        gThrow(lLog.c_str());
    }
    return lProgram;
}

SArr<cl_device_id> sOpenCL::gDevicesID() {
    cl_int err;
    // The `cl_platform_id` structure identifies the
    // first platform identified by the OpenCL runtime.
    // A platform identifies a vendor's installation, so
    // a system may have an NVIDIA platform and an AMD platform.
    SArr<cl_platform_id> platforms;
    {
        cl_uint received_entries = 0;
        err = clGetPlatformIDs(0, NULL, &received_entries);
        gCheckCode(err, "Couldn't count the platforms");
        platforms.resize(received_entries);
        err = clGetPlatformIDs(received_entries, platforms.data(), NULL);
        gCheckCode(err, "Couldn't identify the platforms");
    }
    
    SArr<cl_device_id> devs;
    for (auto const platform : platforms) {
        cl_uint received_entries = 0;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &received_entries);
        gCheckCode(err, "Couldn't count the devices");
        devs.resize(devs.size() + received_entries);
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, received_entries,
                             devs.data() + devs.size() - received_entries, NULL);
        gCheckCode(err, "Couldn't identify the devices");
    }
    
    return devs;
}

void sOpenCL::gCheckCode(cl_int err, sCStrC msg) {
    if (err < 0) {
        if (msg != NULL)
            perror(msg);
        sCStrC msg_cl = NULL;
        switch (err) {
            case CL_SUCCESS:                            msg_cl = "Success!"; break;
            case CL_DEVICE_NOT_FOUND:                   msg_cl = "Device not found."; break;
            case CL_DEVICE_NOT_AVAILABLE:               msg_cl = "Device not available"; break;
            case CL_COMPILER_NOT_AVAILABLE:             msg_cl = "Compiler not available"; break;
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:      msg_cl = "Memory object allocation failure"; break;
            case CL_OUT_OF_RESOURCES:                   msg_cl = "Out of resources"; break;
            case CL_OUT_OF_HOST_MEMORY:                 msg_cl = "Out of host memory"; break;
            case CL_PROFILING_INFO_NOT_AVAILABLE:       msg_cl = "Profiling information not available"; break;
            case CL_MEM_COPY_OVERLAP:                   msg_cl = "Memory copy overlap"; break;
            case CL_IMAGE_FORMAT_MISMATCH:              msg_cl = "Image format mismatch"; break;
            case CL_IMAGE_FORMAT_NOT_SUPPORTED:         msg_cl = "Image format not supported"; break;
            case CL_BUILD_PROGRAM_FAILURE:              msg_cl = "Program build failure"; break;
            case CL_MAP_FAILURE:                        msg_cl = "Map failure"; break;
            case CL_MISALIGNED_SUB_BUFFER_OFFSET:       msg_cl = "CL_MISALIGNED_SUB_BUFFER_OFFSET"; break;
            case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:  msg_cl = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; break;
            case CL_COMPILE_PROGRAM_FAILURE:            msg_cl = "CL_COMPILE_PROGRAM_FAILURE"; break;
            case CL_LINKER_NOT_AVAILABLE:               msg_cl = "CL_LINKER_NOT_AVAILABLE"; break;
            case CL_LINK_PROGRAM_FAILURE:               msg_cl = "CL_LINK_PROGRAM_FAILURE"; break;
            case CL_DEVICE_PARTITION_FAILED:            msg_cl = "CL_DEVICE_PARTITION_FAILED"; break;
            case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      msg_cl = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"; break;
            
            case CL_INVALID_VALUE:                      msg_cl = "Invalid value"; break;
            case CL_INVALID_DEVICE_TYPE:                msg_cl = "Invalid device type"; break;
            case CL_INVALID_PLATFORM:                   msg_cl = "Invalid platform"; break;
            case CL_INVALID_DEVICE:                     msg_cl = "Invalid device"; break;
            case CL_INVALID_CONTEXT:                    msg_cl = "Invalid context"; break;
            case CL_INVALID_QUEUE_PROPERTIES:           msg_cl = "Invalid queue properties"; break;
            case CL_INVALID_COMMAND_QUEUE:              msg_cl = "Invalid command queue"; break;
            case CL_INVALID_HOST_PTR:                   msg_cl = "Invalid host pointer"; break;
            case CL_INVALID_MEM_OBJECT:                 msg_cl = "Invalid memory object"; break;
            case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    msg_cl = "Invalid image format descriptor"; break;
            case CL_INVALID_IMAGE_SIZE:                 msg_cl = "Invalid image size"; break;
            case CL_INVALID_SAMPLER:                    msg_cl = "Invalid sampler"; break;
            case CL_INVALID_BINARY:                     msg_cl = "Invalid binary"; break;
            case CL_INVALID_BUILD_OPTIONS:              msg_cl = "Invalid build options"; break;
            case CL_INVALID_PROGRAM:                    msg_cl = "Invalid program"; break;
            case CL_INVALID_PROGRAM_EXECUTABLE:         msg_cl = "Invalid program executable"; break;
            case CL_INVALID_KERNEL_NAME:                msg_cl = "Invalid kernel name"; break;
            case CL_INVALID_KERNEL_DEFINITION:          msg_cl = "Invalid kernel definition"; break;
            case CL_INVALID_KERNEL:                     msg_cl = "Invalid kernel"; break;
            case CL_INVALID_ARG_INDEX:                  msg_cl = "Invalid argument index"; break;
            case CL_INVALID_ARG_VALUE:                  msg_cl = "Invalid argument value"; break;
            case CL_INVALID_ARG_SIZE:                   msg_cl = "Invalid argument size"; break;
            case CL_INVALID_KERNEL_ARGS:                msg_cl = "Invalid kernel arguments"; break;
            case CL_INVALID_WORK_DIMENSION:             msg_cl = "Invalid work dimension"; break;
            case CL_INVALID_WORK_GROUP_SIZE:            msg_cl = "Invalid work group size"; break;
            case CL_INVALID_WORK_ITEM_SIZE:             msg_cl = "Invalid work item size"; break;
            case CL_INVALID_GLOBAL_OFFSET:              msg_cl = "Invalid global offset"; break;
            case CL_INVALID_EVENT_WAIT_LIST:            msg_cl = "Invalid event wait list"; break;
            case CL_INVALID_EVENT:                      msg_cl = "Invalid event"; break;
            case CL_INVALID_OPERATION:                  msg_cl = "Invalid operation"; break;
            case CL_INVALID_GL_OBJECT:                  msg_cl = "Invalid OpenGL object"; break;
            case CL_INVALID_BUFFER_SIZE:                msg_cl = "Invalid buffer size"; break;
            case CL_INVALID_MIP_LEVEL:                  msg_cl = "Invalid mip-map level"; break;
            case CL_INVALID_GLOBAL_WORK_SIZE:           msg_cl = "CL_INVALID_GLOBAL_WORK_SIZE"; break;
            case CL_INVALID_PROPERTY:                   msg_cl = "CL_INVALID_PROPERTY"; break;
            case CL_INVALID_IMAGE_DESCRIPTOR:           msg_cl = "CL_INVALID_COMPILER_OPTIONS"; break;
            case CL_INVALID_COMPILER_OPTIONS:           msg_cl = "CL_INVALID_COMPILER_OPTIONS"; break;
            case CL_INVALID_LINKER_OPTIONS:             msg_cl = "CL_INVALID_LINKER_OPTIONS"; break;
            case CL_INVALID_DEVICE_PARTITION_COUNT:     msg_cl = "CL_INVALID_DEVICE_PARTITION_COUNT"; break;
            
            default:                                    msg_cl = "Unknown"; break;
        }
        perror(msg);
        throw std::runtime_error(msg_cl);
    }
}

cl_uint sOpenCL::gGetSpecUInt(cl_device_id d, cl_uint s) {
    cl_uint buf_uint;
    sOpenCL::gCheckCode(clGetDeviceInfo(d, s, sizeof(buf_uint), &buf_uint, NULL));
    return buf_uint;
}

cl_ulong sOpenCL::gGetSpecULong(cl_device_id d, cl_uint s) {
    cl_ulong buf_uint;
    sOpenCL::gCheckCode(clGetDeviceInfo(d, s, sizeof(buf_uint), &buf_uint, NULL));
    return buf_uint;
}

sStr sOpenCL::gGetSpecStr(cl_device_id d, cl_uint s) {
    char buffer[1024];
    sOpenCL::gCheckCode(clGetDeviceInfo(d, s, sizeof(buffer), buffer, NULL));
    return buffer;
}

void sOpenCL::gPrintSpecs(cl_device_id d) {
    printf("Picked device:\n");
    printf("- Name = %s\n", gGetSpecStr(d, CL_DEVICE_NAME).c_str());
    printf("- Vendor = %s\n", gGetSpecStr(d, CL_DEVICE_VENDOR).c_str());
    printf("- Device Version = %s\n", gGetSpecStr(d, CL_DEVICE_VERSION).c_str());
    printf("- Driver Version = %s\n", gGetSpecStr(d, CL_DRIVER_VERSION).c_str());
    printf("\n");
    printf("- Max Clock Frequency = %u\n", gGetSpecUInt(d, CL_DEVICE_MAX_CLOCK_FREQUENCY));
    printf("# Max Compute Units = %u\n", gGetSpecUInt(d, CL_DEVICE_MAX_COMPUTE_UNITS));
    printf("# Bytes in VRAM = %llu\n", gGetSpecULong(d, CL_DEVICE_GLOBAL_MEM_SIZE));
    printf("# Bytes in Cacheline = %llu\n", gGetSpecULong(d, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE));
    printf("# Bytes in Cache = %llu\n", gGetSpecULong(d, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE));
    printf("# Bytes in Constant Buffer = %llu\n", gGetSpecULong(d, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE));
    printf("# Bytes in Local Memory = %llu\n", gGetSpecULong(d, CL_DEVICE_LOCAL_MEM_SIZE));
    printf("\n");
    printf("# Bytes Can Allocate At Once = %llu\n", gGetSpecULong(d, CL_DEVICE_MAX_MEM_ALLOC_SIZE));
    printf("# Bytes Argument Can Take = %llu\n", gGetSpecULong(d, CL_DEVICE_MAX_PARAMETER_SIZE));
    printf("# Bytes Is Smallest Alignemnt = %llu\n", gGetSpecULong(d, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE));
    printf("\n");
    printf("# Max Work Group Size = %llu\n", gGetSpecULong(d, CL_DEVICE_MAX_WORK_GROUP_SIZE));
    printf("# Max Work Item Dims = %llu\n", gGetSpecULong(d, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS));
    
    cl_ulong buf_dims[4] = { 0 };
    gCheckCode(clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(buf_dims), &buf_dims, NULL));
    printf("# Max Work Item Sizes = %llu, %llu, %llu, %llu\n",
           (unsigned long long)buf_dims[0],
           (unsigned long long)buf_dims[1],
           (unsigned long long)buf_dims[2],
           (unsigned long long)buf_dims[3]);
    printf("\n");
}

sStr sOpenCL::gGetName(cl_device_id d) {
    return gGetSpecStr(d, CL_DEVICE_NAME);
}
