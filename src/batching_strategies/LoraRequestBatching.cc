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

/// The custom batching strategy cannot specify which request can be merged 
/// to the specific batch, if one request cannot be batched to current batch,
/// it will surely be batched to the next batch. Based on this implementing rule,
/// we don't need any STL container at all because there is no need to enque and 
/// deque the request. For counting and scheduling here, simple array also works fine.
///
/// `lora_request_manager`: record the number of different lora request. Its index
/// stands for `lora type`
/// `lora_request_manager_idx`: the array of `lora_request_manager` index which is not
/// 0, we just need to iterate this array to get info about manager, avoiding
/// meaningless iteration.
/// `cur idx`: the idx of lora_request_manager_idx, default 1, like `.end()` method 
/// in vector
///
// declare
uint32_t* lora_request_manager;
uint32_t* lora_request_manager_idx;
const uint32_t array_size = 1001;
uint32_t cur_idx;
uint32_t* lora_request_cleaner;

/// @brief check if the request pool has this lora type
/// @param lora_type 
/// @return 
bool query_lora_queue(int lora_type){
  if(lora_request_manager[lora_type]){
    // if exist, not zero
    return true;
  }else{
    return false;
  }
}

uint64_t lora_request_batchsize;
std::string lora_request_batchsize_str;

std::deque<std::pair<int, TRITONBACKEND_Request*>>* delayed_requests;

/// \brief The `BatcherInitialize` will be called during model loading.
/// \param batcher point to a user-defined data structure for custom batching 
/// strategy, however this is a double pointer.
/// \param model the model itself.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatcherInitialize(TRITONBACKEND_Batcher** batcher, TRITONBACKEND_Model* model){
  // initialize the manager
  lora_request_manager = new uint32_t[array_size]();      // 4KB
  lora_request_manager_idx = new uint32_t[array_size]();  // 4KB
  cur_idx = 1;
  delayed_requests = new std::deque<std::pair<int, TRITONBACKEND_Request*>>();

  lora_request_cleaner = new uint32_t[array_size]();      // 4KB

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
  *userp = new uint32_t(*reinterpret_cast<const uint32_t*>(batcher));
  uint32_t maximum_lora_requests = (*reinterpret_cast<uint32_t*>(*userp));   // the remained requests maximum
  uint32_t available_space = maximum_lora_requests;

  // push all delayed requests into the batch queue pool
  for(uint32_t i = 0; i < delayed_requests->size() && i <= maximum_lora_requests; i++){
    auto irequest_pair = delayed_requests->front();
    uint32_t ilora_type = irequest_pair.first;
    // TRITONBACKEND_Request *irequest = irequest_pair.second;

    // move this request from the delayed queue to an array queue
    delayed_requests->pop_front();
    lora_request_manager[ilora_type] += 1;
    available_space -= 1;
  }

  delete reinterpret_cast<uint32_t*>(*userp);
  
  *userp = new uint32_t(available_space);
  return nullptr;  // success
}

bool cmp(int a, int b){
  return a > b;
}

bool judge_include(int lora_type, uint32_t available_space){
  const int N = cur_idx;
  int sorted_slots[N] = {0};
  int label = -1;

  for(uint32_t i = 0; i < cur_idx; i++){
    int type_ = lora_request_manager_idx[i];
    sorted_slots[i] = lora_request_manager[type_];
    if(type_ == lora_type){
      label = sorted_slots[i];
    }
  }

  std::sort(sorted_slots, sorted_slots + N, cmp);

  for(int i = 0; available_space > 0; i++){
    if(available_space - sorted_slots[i] > 0){
      if(sorted_slots[i] == label){
        return true;
      }else{
        available_space -= sorted_slots[i];
      }
    }else{
      break;
    }
  }
  // 1. no available space
  // 2. not enough amount
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
  uint32_t available_space = (*reinterpret_cast<uint32_t*>(userp));
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

  // judge if this queue doesn't exist
  if(!query_lora_queue(lora_type_int)){
    // plugin a new slot
    lora_request_manager_idx[cur_idx] = lora_type_int;
    cur_idx++;
  }

  // push the request to the slot
  lora_request_manager[lora_type_int] += 1;

  // judge if this request should be included to the batch
  // if the queue size < 5, included, or will be assigned to next batch
  if(judge_include(lora_type_int, available_space)){
    *should_include = true;
    available_space -= 1;
    lora_request_cleaner[lora_type_int] = 1;
  }
  else{
    auto delayed_pair = std::make_pair(lora_type_int, request);
    delayed_requests->push_back(delayed_pair);
    *should_include = false;

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
  for(uint32_t i = 0; i < cur_idx; i++){
    int lora_type_ = lora_request_manager_idx[i];
    if(lora_request_cleaner[lora_type_]){
      lora_request_manager[lora_type_] = 0;
    }
  }
  return nullptr;  // success
}

/// \brief `BatcherFinalize` will be called during model unloading.
/// \param batcher point to a user-defined data structure, pointer
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONSERVER_Error*
TRITONBACKEND_ModelBatcherFinalize(TRITONBACKEND_Batcher* batcher){
  delete[] lora_request_manager;
  delete[] lora_request_manager_idx;
  cur_idx = 0;
  delete delayed_requests;
  return nullptr;
}

}  // extern "C"

}}}  // namespace triton::core::volume_batching