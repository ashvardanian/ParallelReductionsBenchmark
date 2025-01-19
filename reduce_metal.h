/**
 *  @date 16/01/2025
 *  @file reduce_metal.hpp
 *  @brief Array reductions in Metal
 *  @author Ash Vardanian
 */

#pragma once

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace ashvardanian::reduce {

struct metal_t {

    static constexpr uint32_t treadgroup_size_k = 256;

    float *const begin_ = nullptr; ///< Start of input array
    float *const end_ = nullptr;   ///< End of input array

    // The total number of floats
    std::size_t size_ = 0;

    // Apple Metal objects
    MTL::Device *device_ = nullptr;
    MTL::CommandQueue *queue_ = nullptr;
    MTL::ComputePipelineState *phase1_ = nullptr;
    MTL::ComputePipelineState *phase2_ = nullptr;

    // GPU buffers
    MTL::Buffer *input_buffer_ = nullptr;
    MTL::Buffer *partials_buffer_ = nullptr;
    MTL::Buffer *output_buffer_ = nullptr;
    MTL::Buffer *input_size_buffer_ = nullptr;
    MTL::Buffer *groups_buffer_ = nullptr;

    // Host pointer to read the final sum
    float *host_output_ = nullptr;

    // Number of threadgroups (computed at runtime)
    uint32_t num_groups_ = 0;

    /**
     *  @brief Constructor: allocates and initializes Metal resources.
     *  @param b Pointer to the start of the float array.
     *  @param e Pointer to one-past-the-end of the float array.
     *
     *  This only runs once, so no repeated overhead on each operator()() call.
     */
    metal_t(float *b, float *e) : begin_(b), end_(e) {
        if (!begin_ || !end_ || begin_ >= end_) {
            // No valid input -> skip initialization
            return;
        }
        size_ = static_cast<std::size_t>(end_ - begin_);

        // 1) Create the default Metal device and command queue
        device_ = MTL::CreateSystemDefaultDevice();
        if (!device_) {
            fprintf(stderr, "Metal not supported on this system.\n");
            return;
        }
        queue_ = device_->newCommandQueue();
        if (!queue_) {
            fprintf(stderr, "Failed to create Metal command queue.\n");
            return;
        }

        // 2) Load the precompiled metallib from file (built via CMake)
        //    Make sure "reduce_metal.metallib" is next to the executable
        //    or provide the correct path as needed.
        NS::Error *error = nullptr;
        auto lib_str = NS::String::string("reduce_metal.metallib", NS::UTF8StringEncoding);
        auto library = device_->newLibraryWithFile(lib_str, &error);
        if (!library) {
            fprintf(stderr, "Failed to load reduce_metal.metallib: %s\n",
                    error ? error->localizedDescription()->utf8String() : "(unknown)");
            return;
        }

        // 3) Extract the two functions (phase1 & phase2)
        auto fn_phase1 = library->newFunction(NS::String::string("reduce_phase1", NS::UTF8StringEncoding));
        auto fn_phase2 = library->newFunction(NS::String::string("reduce_phase2", NS::UTF8StringEncoding));
        if (!fn_phase1 || !fn_phase2) {
            fprintf(stderr, "Metal functions not found in reduce_metal.metallib.\n");
            return;
        }

        // 4) Build pipeline states
        NS::Error *pipeline_err1 = nullptr;
        phase1_ = device_->newComputePipelineState(fn_phase1, &pipeline_err1);
        if (!phase1_) {
            fprintf(stderr, "Failed to create pipeline state for reduce_phase1: %s\n",
                    pipeline_err1 ? pipeline_err1->localizedDescription()->utf8String() : "");
            return;
        }
        NS::Error *pipeline_err2 = nullptr;
        phase2_ = device_->newComputePipelineState(fn_phase2, &pipeline_err2);
        if (!phase2_) {
            fprintf(stderr, "Failed to create pipeline state for reduce_phase2: %s\n",
                    pipeline_err2 ? pipeline_err2->localizedDescription()->utf8String() : "");
            return;
        }

        // Done with function objects
        fn_phase1->release();
        fn_phase2->release();
        library->release();

        // 5) Allocate buffers
        //    - input_size bytes for the float array
        //    - partial sums (one per group)
        //    - single-float output
        //    - input_size uniform
        //    - num_groups uniform

        // Copy the input data
        std::size_t bytes = size_ * sizeof(float);
        input_buffer_ = device_->newBuffer(bytes, MTL::ResourceStorageModeShared);
        std::memcpy(input_buffer_->contents(), begin_, bytes);

        // Determine number of threadgroups
        num_groups_ = static_cast<uint32_t>((size_ + treadgroup_size_k - 1) / treadgroup_size_k);
        if (num_groups_ < 1) { num_groups_ = 1; }

        // partials = num_groups_ floats
        partials_buffer_ = device_->newBuffer(num_groups_ * sizeof(float), MTL::ResourceStorageModeShared);
        std::memset(partials_buffer_->contents(), 0, num_groups_ * sizeof(float));

        // final output = 1 float
        output_buffer_ = device_->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);
        std::memset(output_buffer_->contents(), 0, sizeof(float));
        host_output_ = reinterpret_cast<float *>(output_buffer_->contents());

        // uniform: input_size
        input_size_buffer_ = device_->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
        *reinterpret_cast<uint32_t *>(input_size_buffer_->contents()) = static_cast<uint32_t>(size_);

        // uniform: num_groups_
        groups_buffer_ = device_->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
        *reinterpret_cast<uint32_t *>(groups_buffer_->contents()) = num_groups_;
    }

    /**
     *  @brief Perform the reduction on the GPU and return the final sum.
     *  @return Sum of all floats in [begin_, end_]
     *
     *  This enqueues:
     *    1) reduce_phase1 => compute partial sums
     *    2) reduce_phase2 => sum partials into final scalar
     */
    float operator()() noexcept {
        // Early exit if not properly initialized
        if (!device_ || !phase1_ || !phase2_ || size_ == 0) { return 0.0f; }

        // 1) Dispatch phase1
        auto cmd_buffer1 = queue_->commandBuffer();
        auto encoder1 = cmd_buffer1->computeCommandEncoder();

        encoder1->setComputePipelineState(phase1_);
        encoder1->setBuffer(input_buffer_, 0, 0);
        encoder1->setBuffer(partials_buffer_, 0, 1);
        encoder1->setBuffer(input_size_buffer_, 0, 2);

        // We define threadsPerThreadgroup = 256
        MTL::Size tgroup_size(treadgroup_size_k, 1, 1);
        // total # threads = num_groups_ * 256
        MTL::Size grid_size(num_groups_ * treadgroup_size_k, 1, 1);

        encoder1->dispatchThreads(grid_size, tgroup_size);
        encoder1->endEncoding();
        cmd_buffer1->commit();
        cmd_buffer1->waitUntilCompleted();

        // 2) Dispatch phase2 to sum partials => output_buffer_
        auto cmd_buffer2 = queue_->commandBuffer();
        auto encoder2 = cmd_buffer2->computeCommandEncoder();

        encoder2->setComputePipelineState(phase2_);
        encoder2->setBuffer(partials_buffer_, 0, 0);
        encoder2->setBuffer(output_buffer_, 0, 1);
        encoder2->setBuffer(groups_buffer_, 0, 2);

        // We can let 256 threads sum up these partials
        MTL::Size tgroup_size2(treadgroup_size_k, 1, 1);
        MTL::Size grid_size2(treadgroup_size_k, 1, 1);

        encoder2->dispatchThreads(grid_size2, tgroup_size2);
        encoder2->endEncoding();
        cmd_buffer2->commit();
        cmd_buffer2->waitUntilCompleted();

        // read final scalar
        float sum = *host_output_;
        return sum;
    }

    ~metal_t() {
        if (output_buffer_) output_buffer_->release();
        if (partials_buffer_) partials_buffer_->release();
        if (input_buffer_) input_buffer_->release();
        if (input_size_buffer_) input_size_buffer_->release();
        if (groups_buffer_) groups_buffer_->release();
        if (phase1_) phase1_->release();
        if (phase2_) phase2_->release();
        if (queue_) queue_->release();
        if (device_) device_->release();
    }
};

} // namespace ashvardanian::reduce

#endif
