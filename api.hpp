//
// Created by niranda on 12/12/23.
//

#pragma once

#include <cstdint>
#include <memory>
#include <future>
#include <arrow/api.h>

class Error;
class WorkerHandle;

template<typename T, typename E = Error>
class Result;

namespace collective {

using Status = Result<void>;
using Buffer = arrow::Buffer;
using Offsets = arrow::Int64Array;
using MemoryPool = arrow::MemoryPool;

class CollectiveComm {

  /**
   * Initializes the collective communications.
   * @param pool Pool that will be used to allocate memory for internal arrays and buffers by the underneath impl
   */
  virtual void Initialize(MemoryPool *pool) = 0;

  // The self WorkerHandle
  [[nodiscard]] virtual const WorkerHandle &self() const = 0;

  // WorkerHandles of all workers (including self)
  [[nodiscard]] virtual const std::vector<WorkerHandle> &workers() const = 0;

  // WorkerHandles of all workers (including self)
  [[nodiscard]] virtual size_t num_workers() const = 0;

  /**
  * Collective prefix. This will be used by the implementation to generate a unique ID for each operator call
  */
  struct CollPrefix {
    int32_t prefix;
  };


  ///////////////////////////// Callback Definitions /////////////////////////////

  /**
   * Callback for operations that do not produce output data. The status indicates whether the operation has completed
   * successfully or not.
   * Ex: Barrier
   * NOTE: The cb would be invoked by the communication thread that completes the operation. Therefore, the cb is not
   * expected to perform any lengthy computations.
   */
  using CollCb = std::function<void(Status status)>;

  /**
   * Callback for operations that send a fixed input buffer size and produce ONE data buffer. If the operation is
   * successful, the result will own the data_recv buffer; else it will contain an error.
   * Ex: AllGather, Gather, Broadcast
   *
   * Following assertion holds.
   *    data_send.size() is the same in every worker (where applicable)
   *    data_recv.size() == data_send.size() * comm.num_workers();
   *
   * NOTE: The cb would be invoked by the communication thread that completes the operation. Therefore, the cb is not
   * expected to perform any lengthy computations.
   */
  using CollBufferCb = std::function<void(Result<std::shared_ptr<Buffer>/*data_recv*/> result)>;

  /**
   * Callback for operations with variable input buffer sizes in each worker and produce ONE data buffer. If the
   * operation is successful, the result will own the data_recv buffer and offsets_recv array; else it will contain an
   * error.
   * Ex: AllGatherV, GatherV
   * Buffer from the i'th worker,
   *    buffer_i.size() = offsets_recv[i+1] - offsets_recv[i]
   *    buffer_i data begin = data_recv.data() + offsets_recv[i]
   *
   * Following assertion holds.
   *    offsets_recv.size() == comm.num_workers() + 1
   *    data_recv.size() == offsets_recv[comm.num_workers()] - offsets_recv[0] == sum(data_send.size() in every worker)
   *
   * NOTE: The cb would be invoked by the communication thread that completes the operation. Therefore, the cb is not
   * expected to perform any lengthy computations.
   */
  using CollBufferOffsetCb = std::function<void(Result<std::pair<std::shared_ptr<Buffer>/*data_recv*/,
                                                                 std::shared_ptr<Offsets>/*offsets_recv*/>>)>;

  ///////////////////////////// Callback-based Operator Definitions /////////////////////////////

  /**
   * Workers would be waiting asynchronously until all workers reach the corresponding Barrier operation. When this
   * completes, the callback will be called. Barrier does not produce any data.
   * @param prefix Unique prefix for every invocation
   * @param cb Refer CollCb
   * @return Returns immediately. Status indicates whether the operation was successfully accepted by the implementation
   */
  virtual Status Barrier(CollPrefix prefix, CollCb cb) = 0;

  /**
   * A root worker is broadcasting a buffer to every worker (including itself) [one-to-many]. When this completes, the
   * callback will be called with the received data.
   * @param prefix Unique prefix for every invocation
   * @param data_send Data buffer to be sent. The operator assumes ownership of the buffer and will be released once
   * the operation completes asynchronously.
   * @param root Root worker that sends the data.
   * @param cb Refer CollBufferCb.
   * @param pool Memory pool which will be used to allocate memory for the data_recv buffer.
   * @return Returns immediately. Status indicates whether the operation was successfully accepted by the implementation
   */
  virtual Status Broadcast(CollPrefix prefix,
                           std::shared_ptr<Buffer> data_send,
                           WorkerHandle root,
                           CollBufferCb cb,
                           MemoryPool *pool) = 0;

  /**
   * Every worker is sending a fixed-sized buffer to all workers (including itself) [many-to-many]. When this completes,
   * the callback will be called with the received data.
   * @param prefix Unique prefix for every invocation
   * @param data_send Data buffer to be sent. Every worker SHOULD have the SAME data_send.size(); else, the behavior is
   * undefined. The operator assumes ownership of the buffer and will be released once the operation completes
   * asynchronously.
   * @param cb Refer CollBufferCb.
   * @param pool Memory pool which will be used to allocate memory for the data_recv buffer.
   * @return Returns immediately. Status indicates whether the operation was successfully accepted by the implementation
   */
  virtual Status AllGather(CollPrefix prefix,
                           std::shared_ptr<Buffer> data_send,
                           CollBufferCb cb,
                           MemoryPool *pool) = 0;

  /**
   * Every worker (including `root`) is sending a fixed-sized buffer to the `root` worker [many-to-one]. When this
   * completes, the callback will be called with the received data.
   * @param prefix Unique prefix for every invocation
   * @param data_send Data buffer to be sent. Every worker SHOULD have the SAME data_send.size(); else, the behavior is
   * undefined (Use AllGatherV instead). The operator assumes ownership of the buffer and will be released once the
   * operation completes asynchronously.
   * @param cb Refer CollBufferCb. If comm.self() == root, `data_recv` buffer will be valid; else it will be a `nullptr`
   * @param pool Memory pool which will be used to allocate memory for the data_recv buffer.
   * @return Returns immediately. Status indicates whether the operation was successfully accepted by the implementation
   */
  virtual Status Gather(CollPrefix prefix,
                        std::shared_ptr<Buffer> data_send,
                        WorkerHandle root,
                        CollBufferCb cb,
                        MemoryPool *pool) = 0;

  /**
   * Every worker is sending a variable-sized buffer to all workers (including itself) [many-to-many]. When this
   * completes, the callback will be called with the received data and offsets.
   * @param prefix Unique prefix for every invocation
   * @param data_send Data buffer to be sent. The operator assumes ownership of the buffer and will be released once
   * the operation completes asynchronously.
   * @param cb Refer CollBufferOffsetCb.
   * @param pool Memory pool which will be used to allocate memory for the data_recv buffer.
   * @return Returns immediately. Status indicates whether the operation was successfully accepted by the implementation
   */
  virtual Status AllGatherV(CollPrefix prefix,
                            std::shared_ptr<Buffer> data_send,
                            CollBufferOffsetCb cb,
                            MemoryPool *pool) = 0;

  ///////////////////////////// Future-based Operator Definitions /////////////////////////////

  /**
   * Workers would be waiting asynchronously until all workers reach the corresponding Barrier operation. When the
   * operation completes asynchronously, the future will become valid. Barrier does not produce any data.
   * @param prefix Unique prefix for every invocation
   * @return Returns immediately with a future for a Status. If the operation was rejected/unsuccessful, an Error is
   * set for Status
   */
  virtual std::future<Status> Barrier(CollPrefix prefix) = 0;

  /**
   * A root worker is broadcasting a buffer to every worker (including itself) [one-to-many]. When the operation
   * completes asynchronously, the future will become valid.
   * @param prefix Unique prefix for every invocation
   * @param data_send Data buffer to be sent. The operator assumes ownership of the buffer and will be released once
   * the operation completes asynchronously.
   * @param root Root worker that sends the data.
   * @param pool Memory pool which will be used to allocate memory for the data_recv buffer.
   * @return  Returns immediately with a future for a `data_recv` buffer Result. If the operation was
   * rejected/unsuccessful, an Error is set in the Result.
   */
  virtual std::future<Result<std::shared_ptr<Buffer>>> Broadcast(CollPrefix prefix,
                                                                 std::shared_ptr<Buffer> data_send,
                                                                 WorkerHandle root,
                                                                 MemoryPool *pool) = 0;

  /**
   * Every worker is sending a fixed-sized buffer to all workers (including itself) [many-to-many]. When the operation
   * completes asynchronously, the future will become valid.
   * @param prefix Unique prefix for every invocation
   * @param data_send Data buffer to be sent. Every worker SHOULD have the SAME data_send.size(); else, the behavior is
   * undefined. The operator assumes ownership of the buffer and will be released once the operation completes
   * asynchronously.
   * @param pool Memory pool which will be used to allocate memory for the data_recv buffer.
   * @return Returns immediately with a future for a `data_recv` buffer Result. If the operation was
   * rejected/unsuccessful, an Error is set in the Result.
   */
  virtual std::future<Result<std::shared_ptr<Buffer>>> AllGather(CollPrefix prefix,
                                                                 std::shared_ptr<Buffer> data_send,
                                                                 MemoryPool *pool) = 0;

  /**
   * Every worker (including `root`) is sending a fixed-sized buffer to the `root` worker [many-to-one]. When the
   * operation completes asynchronously, the future will become valid.
   * @param prefix Unique prefix for every invocation
   * @param data_send Data buffer to be sent. Every worker SHOULD have the SAME data_send.size(); else, the behavior is
   * undefined (Use AllGatherV if this cannot b guaranteed). The operator assumes ownership of the buffer and will
   * be released once the operation completes asynchronously.
   * @param pool Memory pool which will be used to allocate memory for the data_recv buffer.
   * @return Returns immediately with a future for a `data_recv` buffer Result. If the operation was
   * rejected/unsuccessful, an Error is set in the Result. If comm.self() == root, `data_recv` buffer will be valid;
   * else it will be a `nullptr`.
   */
  virtual std::future<Result<std::shared_ptr<Buffer>>> Gather(CollPrefix prefix,
                                                              std::shared_ptr<Buffer> data_send,
                                                              WorkerHandle root,
                                                              MemoryPool *pool) = 0;

  /**
   * Every worker is sending a variable-sized buffer to all workers (including itself) [many-to-many].  When the
   * operation completes asynchronously, the future will become valid.
   * @param prefix Unique prefix for every invocation
   * @param data_send Data buffer to be sent. The operator assumes ownership of the buffer and will be released once
   * the operation completes asynchronously.
   * @param pool Memory pool which will be used to allocate memory for the data_recv buffer.
   * @return Returns immediately with a future for a `data_recv` buffer & an `offsets` array pair Result. If the
   * operation was rejected/unsuccessful, an Error is set in the Result.
   */
  virtual std::future<Result<std::pair<std::shared_ptr<Buffer>,
                                       std::shared_ptr<Offsets>>>> AllGatherV(CollPrefix prefix,
                                                                              std::shared_ptr<Buffer> data_send,
                                                                              MemoryPool *pool) = 0;
};

} // namespace collective