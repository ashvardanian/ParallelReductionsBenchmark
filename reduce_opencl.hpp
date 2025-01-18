/**
 *  @date 04/09/2019
 *  @file reduce_opencl.hpp
 *  @brief OpenCL host code for parallel reductions
 *  @author Ash Vardanian
 */
#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <fmt/core.h> // `fmt::format`

#define CL_SILENCE_DEPRECATION 1
#if defined(__APPLE__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace ashvardanian::reduce {

struct opencl_target_t {
    std::string device_name;
    std::string device_version;
    std::string driver_version;
    std::string language_version;

    cl_platform_id platform;
    cl_device_id device;
    cl_uint compute_units;
};

static int opencl_wg_sizes[] = {64, 128, 256, 512, 1024};
static int const opencl_max_threads = 12000;

inline char const *opencl_error_name(cl_int);
inline std::vector<opencl_target_t> opencl_targets();

struct opencl_t {

    static constexpr size_t kernel_variants_k = 8;
    static constexpr char const *kernels_k[kernel_variants_k] = {
        "reduce_simple",  "reduce_w_modulo", "reduce_in_shared",      "reduce_w_sequential_addressing",
        "reduce_bi_step", "reduce_unrolled", "reduce_unrolled_fully", "reduce_w_brents_theorem",
    };

    size_t const count_items = 0;
    size_t const count_threads = 0;
    size_t const items_per_group = 0;

  private:
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;

    /**
     * The main dataset pre-copied to target device.
     */
    cl_mem dataset = NULL;
    /**
     * Global memory for partial sums outputs.
     * Size: |threads| * sizeof(float).
     */
    cl_mem global_outputs = NULL;
    std::vector<float> returned_outputs;

  public:
    opencl_t(float const *b, float const *e, opencl_target_t target, size_t items_per_group_ = 1024,
             char const *kernel_name_cstr = kernels_k[0])
        : count_items(e - b), count_threads((opencl_max_threads / items_per_group_) * items_per_group_),
          items_per_group(items_per_group_) {
        // Load the kernel source code into the array source_str
        std::string source_str;
        {
            std::ifstream t("../reduce_opencl.cl");
            if (!t.is_open())
                throw std::logic_error("Could not open file\n");
            std::stringstream buffer;
            buffer << t.rdbuf();
            source_str = buffer.str();
        }

        {
            size_t max_work_group_size = 0;
            clGetDeviceInfo(target.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
            if (max_work_group_size < items_per_group_)
                throw std::logic_error(fmt::format("Max work group size: {} ====> Given work group size: {}\n",
                                                   max_work_group_size, items_per_group_));
        }

        cl_int status = 0;
        context = clCreateContext(NULL, 1, &target.device, NULL, NULL, &status);

        // Create a command queue
        queue = clCreateCommandQueue(context, target.device, 0, &status);

        // Create memory buffers on the device for each vector
        // https://www.khronos.org/registry/OpenCL/sdk/2.2/docs/man/html/clCreateBuffer.html
        dataset = clCreateBuffer(context, CL_MEM_READ_ONLY, count_items * sizeof(float), NULL, &status);
        global_outputs =
            clCreateBuffer(context, CL_MEM_READ_WRITE,
                           ((count_items + items_per_group_ - 1) / items_per_group_) * sizeof(float), NULL, &status);
        returned_outputs.resize(count_threads);

        // Move the `dataset` to GPU.
        // We don't need to explicitly finish the queue, as this transfer is blocking.
        // https://www.khronos.org/registry/OpenCL/sdk/2.2/docs/man/html/clEnqueueReadBuffer.html
        status = clEnqueueWriteBuffer(queue, dataset, CL_TRUE, 0, count_items * sizeof(float), b, 0, NULL, NULL);

        // Create a program from the kernel source
        char const *source_cstr = source_str.c_str();
        size_t const source_size = source_str.size();
        program = clCreateProgramWithSource(context, 1, &source_cstr, &source_size, &status);

        // The third parameter is the list of devices.
        // If it's NULL, the program executable is built for all devices
        // associated with program for which a source or binary has been loaded.
        //
        // The fourth parameter accepts options that configure the compilation.
        // These are similar to the flags used by gcc. For example, you can
        // define a macro with the option -DMACRO=VALUE and turn off optimization
        // with -cl-opt-disable.
        //
        // Docs: https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clBuildProgram.html
        status = clBuildProgram(program, 1, &target.device, NULL, NULL, NULL);

        // Create the OpenCL kernel
        kernel = clCreateKernel(program, kernel_name_cstr, &status);

        // Set the arguments of the kernel
        auto local_buffers_size = items_per_group * sizeof(float);
        status = clSetKernelArg(kernel, 0, sizeof(dataset), &dataset);
        status = clSetKernelArg(kernel, 1, sizeof(global_outputs), &global_outputs);
        status = clSetKernelArg(kernel, 2, local_buffers_size, NULL);

        if (status != 0)
            throw std::logic_error(opencl_error_name(status));
    }

    ~opencl_t() {
        cl_int status = 0;
        status = clFlush(queue);
        status = clFinish(queue);

        status = clReleaseMemObject(dataset);
        status = clReleaseMemObject(global_outputs);
        status = clReleaseKernel(kernel);
        status = clReleaseProgram(program);
        status = clReleaseCommandQueue(queue);
        status = clReleaseContext(context);

        if (status != 0)
            // We probably shouldn't throw in the destructor :)
            //     throw std::logic_error(opencl_error_name(status));
            (void)status;
    }

    float operator()() {
        cl_int status = 0;
        size_t global_ws_offset = 0;
        status =
            clEnqueueNDRangeKernel(queue, kernel, 1, &global_ws_offset, &count_items, &items_per_group, 0, NULL, NULL);
        if (status != 0)
            throw std::logic_error(opencl_error_name(status));
        status = clFlush(queue);

        // We don't need to explicitly finish the queue, as this transfer is blocking.
        // https://www.khronos.org/registry/OpenCL/sdk/2.2/docs/man/html/clEnqueueReadBuffer.html
        status = clEnqueueReadBuffer(queue, global_outputs, CL_TRUE, 0, returned_outputs.size() * sizeof(float),
                                     returned_outputs.data(), 0, NULL, NULL);

        if (status != 0)
            throw std::logic_error(opencl_error_name(status));

        return returned_outputs.front();
    }
};

inline std::vector<opencl_target_t> opencl_targets() {

    std::vector<opencl_target_t> result;
    size_t string_length;
    cl_uint platform_count;
    cl_uint device_count;

    clGetPlatformIDs(0, NULL, &platform_count);
    std::vector<cl_platform_id> platforms(platform_count);
    clGetPlatformIDs(platform_count, platforms.data(), NULL);

    for (auto platform : platforms) {

        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &device_count);
        std::vector<cl_device_id> devices(device_count);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, device_count, devices.data(), NULL);

        for (auto device : devices) {

            opencl_target_t target;
            target.platform = platform;
            target.device = device;
            clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(target.compute_units), &target.compute_units,
                            NULL);

            // Extract the variable length string descriptors.
            clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &string_length);
            target.device_name.resize(string_length);
            clGetDeviceInfo(device, CL_DEVICE_NAME, string_length, (void *)target.device_name.data(), NULL);

            clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &string_length);
            target.device_version.resize(string_length);
            clGetDeviceInfo(device, CL_DEVICE_VERSION, string_length, (void *)target.device_version.data(), NULL);

            clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &string_length);
            target.driver_version.resize(string_length);
            clGetDeviceInfo(device, CL_DRIVER_VERSION, string_length, (void *)target.driver_version.data(), NULL);

            clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &string_length);
            target.language_version.resize(string_length);
            clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, string_length, (void *)target.language_version.data(),
                            NULL);

            result.push_back(target);
        }
    }
    return result;
}

inline char const *opencl_error_name(cl_int code) {
    // clang-format off
    switch (code) {
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
        case -69: return "CL_INVALID_PIPE_SIZE";
        case -70: return "CL_INVALID_DEVICE_QUEUE";
        case -71: return "CL_INVALID_SPEC_ID";
        case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
        case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
        case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
        case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
        case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
        case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
        case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
        case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
        case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
        case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
        case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
        case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
        case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
        case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
        case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
        case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
        case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
        case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
        default: return "CL_UNKNOWN_ERROR";
    }
    // clang-format on
}

} // namespace ashvardanian::reduce