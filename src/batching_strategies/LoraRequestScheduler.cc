#include <stdint.h>

#include <exception>
#include <string>
#include <thread>
#include <vector>
#include <deque>
#include <map>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"

#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"

#define RESPOND_ALL_AND_RETURN_IF_ERROR(RESPONSES, RESPONSES_COUNT, X) \
  do {                                                                 \
    TRITONSERVER_Error* raarie_err__ = (X);                            \
    if (raarie_err__ != nullptr) {                                     \
      SendErrorForResponses(RESPONSES, RESPONSES_COUNT, raarie_err__); \
      return;                                                          \
    }                                                                  \
  } while (false)

namespace triton { namespace core { namespace lora_scheduler {

/////////////

extern "C" {

std::map<std::string, std::deque<TRITONBACKEND_Request*>> * lora_deque_manager;

bool query_lora_queue(std::string lora_type){
  // assume the deque set is not empty
  if(lora_deque_manager.find(lora_type) != lora_deque_manager.end()){
    return true;
  }else{
    return false;
  }
}

void 
push_request(std::string lora_type, TRITONBACKEND_Request* request){
  lora_deque_manager[lora_type].push_back(request);
}

/// \brief The `BatcherInitialize` will be called during model loading.
/// \param batcher point to a user-defined data structure for custom batching 
/// strategy, however this is a double pointer.
/// \param model the model itself.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatcherInitialize(TRITONBACKEND_Batcher** batcher, TRITONBACKEND_Model* model){
  // initialize the manager
  lora_deque_manager = new std::map<std::string, std::deque<TRITONBACKEND_Request*>>();

  // extract the parameters
  TRITONSERVER_Message* config_message;
  TRITONBACKEND_ModelConfig(model, 1 /* config_version */, &config_message);

  const char* buffer;
  size_t byte_size;

  uint64_t max_queue_delay = 0;
  std::string max_queue_delay_str;

  auto err =
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size);
  if (err)
    return err;

  triton::common::TritonJson::Value model_config, dynamic_batching_params, delay_param;
  err = model_config.Parse(buffer, byte_size);
  TRITONSERVER_MessageDelete(config_message);

  if (!model_config.Find("dynamic_batching", &dynamic_batching_params)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        "Unable to find `dynamic_batching` in model config");
  }

  std::vector<std::string> param_keys;

  if (!params.Find("max_queue_delay_microseconds", &delay_param)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        "Unable to find `max_queue_delay_microseconds` parameter in model config");
  }

  err = delay_param.MemberAsString("string_value", &max_queue_delay_str);
  if (err)
    return err;

  try{
    max_queue_delay = static_cast<uint64_t>(std::stoul(max_queue_delay_str));
  }
  catch (const std::invalid_argument& ia) {
    return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("failed to convert '") + max_queue_delay_str +
         "' to unsigned int64").c_str()
    )
  }

  *batcher = reinterpret_cast<TRITONBACKEND_Batcher*>(
      new unsigned int(max_queue_delay));
  return nullptr;   // success
}

/// Callback to be invoked when Triton has begun forming a batch.
///
/// \param batcher The read-only placeholder for backend to retrieve
// information about the batching strategy for this model.
/// \param userp The placeholder for backend to store and retrieve information
/// about this pending batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatchInitialize(
    const TRITONBACKEND_Batcher* batcher, void** userp)
{
  // Userp will point to an unsigned integer representing the remaining volume
  // in bytes for this batch.
  *userp = new unsigned int(*reinterpret_cast<const unsigned int*>(batcher));
  return nullptr;  // success
}

/// Check whether a request should be added to the pending model batch.
///
/// \param request The request to be added to the pending batch.
/// \param userp The placeholder for backend to store and retrieve information
/// about this pending batch. When the callback returns, this should reflect
/// the latest batch information.
/// \param should_include The pointer to be updated on whether the request
/// should be included in the batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatchIncludeRequest(
    TRITONBACKEND_Request* request, void* userp, bool* should_include)
{
  // Default should_include to false in case function returns error.
  *should_include = false;

  uint32_t input_count;
  auto err = TRITONBACKEND_RequestInputCount(request, &input_count);
  if (err)
    return err;

  TRITONBACKEND_Input* input;
  const char* property_name;

  for (size_t count = 0; count < input_count; count++) {
    // fetch the content according to the property count.

    // fetch single input
    auto err =
        TRITONBACKEND_RequestInputByIndex(request, count /* index */, &input);
    if (err)
      return err;

    // fetch the property name
    err = TRITONBACKEND_InputProperties(
        input, &property_name, nullptr, nullptr, nullptr, nullptr, nullptr);
    if (err)
      return err;
    
    if (strcmp(property_name, "lora_type") == 0){
      *should_include = true;
      
      void* property_buffer;
      uint64_t property_buffer_size;
      TRITONSERVER_MemoryType property_memory_type;
      int64_t property_memory_type_id;
      err = TRITONBACKEND_InputBuffer(input, 0, &property_buffer, &property_buffer_size, 
                                      &property_memory_type, &property_memory_type_id);

      char* lora_type = reinterpret_cast<char*> property_buffer;
      std::string lora_type_str(lora_type);

      // process logic ...
      if(!query_lora_queue(lora_type_str)){
        std::deque<TRITONBACKEND_Request*> req_queue;
        std::pair<std::string, std::deque<TRITONBACKEND_Request*>> pair_buffer = {lora_type_str, req_queue};
        lora_deque_manager.insert(pair_buffer);
      }
      lora_deque_manager[lora_type_str].push_back(request);


      break;
    } else {
      continue;
    }
  }

  return nullptr;  // success
}

/// Callback to be invoked when Triton has finishing forming a batch.
///
/// \param userp The placeholder for backend to store and retrieve information
/// about this pending batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatchFinalize(void* userp)
{
  delete static_cast<unsigned int*>(userp);
  return nullptr;  // success
}

/// \brief `BatcherFinalize` will be called during model unloading.
/// \param batcher point to a user-defined data structure, pointer
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatcherFinalize(TRITONBACKEND_Batcher* batcher){
  delete lora_deque_manager;

  return nullptr;
}

}  // extern "C"

}}}  // namespace triton::core::volume_batching