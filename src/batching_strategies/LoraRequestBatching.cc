#include <stdint.h>

#include <exception>
#include <string>
#include <thread>
#include <deque>
#include <map>
#include <algorithm>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"

#define RESPOND_ALL_AND_RETURN_IF_ERROR(RESPONSES, RESPONSES_COUNT, X) \
  do {                                                                 \
    TRITONSERVER_Error* raarie_err__ = (X);                            \
    if (raarie_err__ != nullptr) {                                     \
      SendErrorForResponses(RESPONSES, RESPONSES_COUNT, raarie_err__); \
      return;                                                          \
    }                                                                  \
  } while (false)

namespace triton { namespace core { namespace lora_batching {

/////////////

extern "C" {

/// Every in-coming lora request with an identifier could be recognized,
/// all we need to do is to make requests with the same identifier batched
/// as more as possible. If the request's lora type is a rare one, then we 
/// will delay it to next batch.
///
/// We use `map` to store the queue of different lora type. The type of deque
/// is bool, to symbolize whether a request should be batched in current
/// batch or not. We don't use TRITONBACKEND_Request* because it's meaningless
/// for the scheduling algorithm here. Using bool type directly can reduce code
/// operation and process more logically.
///
/// When a request comes, it will be judged to be into the current batch. If 
/// the total number of all queues is smaller than the required amount, then 
/// true, otherwise false.
///
/// In batch_finalize API function, all the `true` elements will pop from the
/// queue, and all the `false` elements will be converted to true, because they 
/// are delayed to the next batch and must be true.
///

std::map<int, std::deque<bool>> *request_pool;

/// \brief check if the request pool has this lora type
/// \param lora_type integer
/// \return 
bool query_lora_queue(int lora_type){
  if(request_pool->count(lora_type)){
    return true;
  }else{
    return false;
  }
}

uint64_t lora_request_batchsize;
std::string lora_request_batchsize_str;

/// \brief The `BatcherInitialize` will be called during model loading.
/// \param batcher point to a user-defined data structure for custom batching 
/// strategy, however this is a double pointer.
/// \param model the model itself.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatcherInitialize(TRITONBACKEND_Batcher** batcher, TRITONBACKEND_Model* model){
  // initialize the request pool
  request_pool = new std::map<int, std::deque<bool>>();

  // extract the parameters
  TRITONSERVER_Message* config_message;
  TRITONBACKEND_ModelConfig(model, 1 /* config_version */, &config_message);

  const char* buffer;
  size_t byte_size;

  lora_request_batchsize = 0;


  auto err =
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size);
  if (err)
    return err;

  triton::common::TritonJson::Value model_config, params, batchsize_param;
  err = model_config.Parse(buffer, byte_size);
  TRITONSERVER_MessageDelete(config_message);

  if (!model_config.Find("parameters", &params)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        "Unable to find `parameters` in model config");
  }

  if (!params.Find("lora_request_batchsize", &batchsize_param)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        "Unable to find `lora_request_batchsize` parameter in model config");
  }

  err = batchsize_param.MemberAsString("string_value", &lora_request_batchsize_str);
  if (err)
    return err;

  try{
    lora_request_batchsize = static_cast<uint64_t>(std::stoul(lora_request_batchsize_str));
  }
  catch (const std::invalid_argument& ia) {
    return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("failed to convert '") + lora_request_batchsize_str +
         "' to unsigned int64").c_str()
    );
  }

  *batcher = reinterpret_cast<TRITONBACKEND_Batcher*>(
      new uint64_t(lora_request_batchsize));
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
  // Userp will first point to the batcher indicating the maximum of the lora request
  // and will sub according to the delayed_queue requests number.
  uint32_t* maximum_lora_requests = new uint32_t(*reinterpret_cast<const uint32_t*>(batcher));  // the remained requests maximum
  uint32_t available_space = *maximum_lora_requests;

  for(auto ite = request_pool->begin(); ite != request_pool->end(); ite++){
    auto& _queue = ite->second;
    if(!_queue.empty()){
      for(int i = 0; i<_queue.size(); i++){
        bool& boolean = _queue.at(i);
        if(!boolean){
          boolean = true;
        }
        if(available_space > 0)
          available_space--;
      }
    }
  }

  
  *userp = new uint32_t(available_space);
  return nullptr;  // success
}

bool judge_include(int lora_type, uint32_t available_space){
  // judge if this queue doesn't exist
  if(!query_lora_queue(lora_type)){
    // plugin a new slot
    std::deque<bool> new_queue;
    new_queue.push_back(false);
    std::pair<int, std::deque<bool>> new_slot(lora_type, new_queue);
    request_pool->insert(new_slot);
    return false;
  }

  /// now the slot already exist.
  auto _pair = request_pool->find(lora_type);
  auto& _queue = _pair->second;
  if(available_space > lora_request_batchsize/2){
    _queue.push_back(true);
    return true;
  }
  else{
    // follow the last element, if null then false.
    if(_queue.empty()){
      _queue.push_back(false);
      return false;
    }
    else{
      bool follower = _queue.back();
      _queue.push_back(follower);
      return follower;
    }
  }
  return false;
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
  uint32_t *available_space = static_cast<uint32_t*>(userp);
  // uint32_t preferred_pipeline = 4;

  uint32_t input_count;
  auto err = TRITONBACKEND_RequestInputCount(request, &input_count);
  if (err)
    return err;

  TRITONBACKEND_Input* input;
  const char* property_name;
  int lora_type_int = -1;

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
      
      const void* property_buffer;
      uint64_t property_buffer_size;
      TRITONSERVER_MemoryType property_memory_type;
      int64_t property_memory_type_id;
      err = TRITONBACKEND_InputBuffer(input, 0, &property_buffer, &property_buffer_size, 
                                      &property_memory_type, &property_memory_type_id);

      const char* lora_type = reinterpret_cast<const char*>(property_buffer);
      lora_type_int = std::atoi(lora_type);
      LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Lora Type: ") + std::string(lora_type)).c_str()
      );
      break;
    } else 
      continue;
  }
  // process logic:
  
  *should_include = judge_include(lora_type_int, *available_space) && *available_space;
  if(*should_include){
    *available_space -= 1;
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
  for(auto ite = request_pool->begin(); ite != request_pool->end(); ite++){
    // pull all `true` out of the queue
    auto& _queue = ite->second;
    while(_queue.front()){
      _queue.pop_front();
    }
  }
  delete static_cast<unsigned int*>(userp);
  return nullptr;  // success
}

/// \brief `BatcherFinalize` will be called during model unloading.
/// \param batcher point to a user-defined data structure, pointer
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatcherFinalize(TRITONBACKEND_Batcher* batcher){
  delete request_pool;

  return nullptr;
}

}  // extern "C"

}}}  // namespace triton::core::volume_batching