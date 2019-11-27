/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {

using TaskId = size_t;

// Record, containing tensor size/shape and IDs of the first and the last task,
// that use this tensor as input or output. For example: tensor #3 with size
// tensor_size=65536 is first introduced in program #2 (first_task=2) and used
// for the last time in program #7 (last_task=7).
template <typename TensorSizeT>
struct TensorUsageRecord {
  TensorSizeT tensor_size;
  TaskId first_task;
  TaskId last_task;

  TensorUsageRecord(TensorSizeT size, TaskId first, TaskId last)
      : tensor_size(size), first_task(first), last_task(last) {}

  // Default order of tensor usage records is increasing order of first_task.
  bool operator<(const TensorUsageRecord<TensorSizeT>& other) const {
    return first_task < other.first_task;
  }
};

// Information about assignment of tensors to shared objects
template <typename TensorSizeT>
struct ObjectsAssignment {
  // shared_object_ids_[i] is ID of shared object, that tensor i will be using.
  std::vector<size_t> object_ids;
  // shared_object_sizes_[i] is a size of shared object with ID equal to i.
  std::vector<TensorSizeT> object_sizes;
};

enum class MemoryStrategy {
  // Naive strategy is to allocate each object separately.
  // Can be useful for debugging to see all intermediate outputs.
  NAIVE,

  // Equality strategy allows to reuse the same part of memory for several
  // tensors with the same size, but non-intersecting usage intervals.
  EQUALITY,

  // Greedy strategy uses greedy algorithm to reuse memory from tensors, that
  // won't be used anymore, for new ones.
  GREEDY,

  // Mincostflow strategy consists of building auxiliary flow graph and solving
  // the minimum-cost flow problem in it. In the end edges with zero residual
  // capacity determine assignment of shared objects to tensors.
  MINCOSTFLOW,
};

// Calculates the assignement of shared objects to given tensors, including
// objects' sizes. Initial tensor sizes are given as size_t. This function is
// intended to use with GPU buffers.
Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    const MemoryStrategy& strategy, ObjectsAssignment<size_t>* assignment);

// Calculates the assignement of shared objects to given tensors, including
// objects' sizes. Initial tensor sizes are given as BHWC. This function is
// intended to use with GPU textures.
Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<BHWC>>& usage_records,
    const MemoryStrategy& strategy, ObjectsAssignment<BHWC>* assignment);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MEMORY_MANAGEMENT_H_
