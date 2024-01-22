#include <vector>
#include "benchmark/core/model.h"
#include "benchmark/proto/sample.pb.h"

#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"
#include "absl/synchronization/notification.h"
#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/common_runtime/device_factory.h"

using namespace tensorflow;
namespace benchmark {

Model::PredictContext *Model::Borrow() {
  std::unique_lock<std::mutex> lock(mutex_);
  for (auto& context : predict_contexts_) {
    if (!context.borrowed) {
      context.borrowed = true;
      return &context;
    }
  }
  auto timeout = std::chrono::microseconds(500);
  PredictContext *res = nullptr;
  auto pred = [&]() {
    for (auto& context : predict_contexts_) {
      if (!context.borrowed) {
        context.borrowed = true;
        res = &context;
        return true;
      }
    }
    return false;
  };
  if (cond_.wait_for(lock, timeout, pred)) {
    return res;
  }
  return nullptr;
}

void Model::Return(PredictContext *predict_context) {
  std::unique_lock<std::mutex> lock(mutex_);
  for (auto& context : predict_contexts_) {
    if (context.session == predict_context->session) {
      context.borrowed = false;
    }
  }
  cond_.notify_one();
}

bool Model::ParseRunOptions(const std::string& run_options) {
  if (run_options.empty()) {
    LOG(WARNING) << "No run_options path configured: " << name()
                 << ", use default RunOptions.";
    return false;
  }
  Status s = ReadTextProto(Env::Default(), run_options.c_str(), &run_options_);
  if (!s.ok()) {
    s = ReadBinaryProto(Env::Default(), run_options.c_str(), &run_options_);
    if (!s.ok()) {
      LOG(ERROR) << "Read run options failed: " << name() << ", " << s.ToString();
      return false;
    }
  }
  VLOG(1) << "Read run options, " << run_options << ": " << run_options_.DebugString();
  return true;
}

bool Model::InitSession(const std::string& config_proto) {

  if (inputs_.empty()) {
    LOG(ERROR) << "No vaild inputs: " << name();
    return false;
  }

  // Prepare Session ConfigProto
  Status s;
  SessionOptions session_options = SessionOptions();
  ConfigProto* config = &session_options.config;
  
  /*
  GPUOptions* gpu_options = config->mutable_gpu_options();
  auto virtual_devices = gpu_options->mutable_experimental()->add_virtual_devices();
  float per_virtual_device_memory = 15109 / this->predictor_num_;
  for (int i = 0; i < this->predictor_num_; ++i) {
    virtual_devices->add_memory_limit_mb(per_virtual_device_memory);
  }

  std::vector<std::unique_ptr<Device>> devices;
  DeviceFactory::GetFactory("GPU")->CreateDevices(
      session_options, "", &devices);

  std::cout << "list devices" << std::endl;
  for (int i = 0; i < devices.size(); ++i) {
    std::cout << "device: " << devices[i]->name() << std::endl;
  }
  */

  std::string xla_config = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit";
  setenv("TF_XLA_FLAGS", xla_config.c_str(), true /*overwrite*/);

  tensorflow::GraphOptions graphOptions;
  tensorflow::OptimizerOptions optimizerOptions;
  optimizerOptions.set_do_constant_folding(true);
  optimizerOptions.set_do_function_inlining(true);
  optimizerOptions.set_do_common_subexpression_elimination(true);
    // graphOptions.set_allocated_optimizer_options(&optimizerOptions);
  *graphOptions.mutable_optimizer_options() = optimizerOptions;
  *(config->mutable_graph_options()) = graphOptions;
    // 2.  设置默认线程数 对于小模型多线程反而消耗io时间
    // 2.1 线程数设置为1时性能最好 可以和python对齐 5ms
  config->set_intra_op_parallelism_threads(1);
  config->set_use_per_session_threads(true);
  config->set_inter_op_parallelism_threads(1);


  if (!config_proto.empty()) {
    s = ReadTextProto(Env::Default(), config_proto.c_str(), config);
    if (!s.ok()) {
      s = ReadBinaryProto(Env::Default(), config_proto.c_str(), config);
      if (!s.ok()) {
        LOG(ERROR) << "Read config proto failed: " << name() << ", " << s.ToString()
                   << ". Use default ConfigProto.";
      }
    }
    VLOG(1) << "Read config proto: " << name() << ", " << config->DebugString();
  }
  config->set_allow_soft_placement(true);

  for (int i = 0; i < this->predictor_num_; ++i) {

    Session* session = nullptr;
    std::string session_key = name() + "/session:" + std::to_string(i);
    auto iter = sessions_.find(session_key);
    if (iter == sessions_.end()) {
      s = NewSession(session_options, &session);
      if (!s.ok()) {
        LOG(ERROR) << "New session failed: " << name() << ", " << s.ToString();
        return false;
      }
      s = session->Create(gdef_);
      if (!s.ok()) {
        LOG(ERROR) << "Create session failed: " << name() << ", " << s.ToString();
        return false;
      }
      sessions_[session_key] = session;
    } else {
      session = iter->second;
    }
    PredictContext context{session, false, this};
    predict_contexts_.push_back(context);
    LOG(INFO) << "Predictor " << i << " uses session " << session_key;
  }
  return true;
}

Tensor ConvertToBfloat16(const Tensor& input_tensor) {
  // Create a new tensor with float16 data type and the same shape as the input tensor
  Tensor bfloat16_tensor(DT_BFLOAT16, input_tensor.shape());

  // Get pointers to the input and output data
  const float* input_data = input_tensor.flat<float>().data();
  bfloat16* bfloat16_data = bfloat16_tensor.flat<bfloat16>().data();

  // Convert each element from float32 to float16
  const int num_elements = input_tensor.NumElements();
  for (int i = 0; i < num_elements; ++i) {
    bfloat16_data[i] = bfloat16(input_data[i]);
  }

  return bfloat16_tensor;
}

Tensor ConvertToFloat16(const Tensor& input_tensor) {
  Tensor float16_tensor(DT_HALF, input_tensor.shape());
  // Get pointers to the input and output data
  const float* input_data = input_tensor.flat<float>().data();
  Eigen::half* float16_data = float16_tensor.flat<Eigen::half>().data();

  // Convert each element from float32 to float16
  const int num_elements = input_tensor.NumElements();
  for (int i = 0; i < num_elements; ++i) {
    float16_data[i] = Eigen::half(input_data[i]);
  }

  return float16_tensor;
}

bool Model::ParseSamples(const std::string& sample_file) {
  if (sample_file.empty()) {
    LOG(ERROR) << "Samplefile path must not be empty: " << name();
    return false;
  }
  SamplesProto samples_proto;
  std::cout << "begin read binary proto" << std::endl;
  Status s = ReadBinaryProto(Env::Default(), sample_file.c_str(), &samples_proto);
  if (!s.ok()) {
    s = ReadTextProto(Env::Default(), sample_file.c_str(), &samples_proto);
    if (!s.ok()) {
      LOG(ERROR) << "Read sample_file failed: " << name() << ", " << s.ToString();
      return false;
    }
  }
  VLOG(2) << "Samples_proto: " << samples_proto.DebugString();
  std::cout << "end read binary proto" << std::endl;

  for (int i = 0; i < samples_proto.output_names_size(); ++i) {
    output_names_.push_back(samples_proto.output_names(i));
  }
  
  inputs_.clear();
  inputs_.reserve(samples_proto.sample_size());

  for (int i = 0; i < samples_proto.sample_size(); ++i) {
    const InputsProto& inputs_proto = samples_proto.sample(i);
    std::vector<std::pair<std::string, Tensor>> sample_vec;
    int64 batchsize = 1;
    for (int j = 0; j < inputs_proto.input_size(); ++j) {
      const NamedTensorProto& input = inputs_proto.input(j);
      Tensor tensor;
      if (!tensor.FromProto(input.tensor())) { 
        LOG(ERROR) << "Init tensor from proto failed.";
        return false;
      }

      std::cout << "before convert tensor to float16";
      Tensor float16_tensor = ConvertToFloat16(tensor); 
      sample_vec.emplace_back(input.name(), float16_tensor);
      if (tensor.dims() >=1 && tensor.dim_size(0) > batchsize)
        batchsize = tensor.dim_size(0);
    }
    inputs_.emplace_back(std::move(sample_vec));
    inferred_batchsizes_.emplace_back(batchsize);
    LOG(INFO) << "Parsed input, inferred batchsize = " << batchsize;
  }

  LOG(INFO) << "Parse input_samples success, total "<<inputs_.size()<<"samples";

  return true;
}

bool Model::LoadGraph(const std::string& frozen_graph) {
  if (frozen_graph.empty()) {
    LOG(ERROR) << "No graph path configured: " << name();
    return false;
  }
  Status s = ReadTextProto(Env::Default(), frozen_graph.c_str(), &gdef_);
  if (!s.ok()) {
    s = ReadBinaryProto(Env::Default(), frozen_graph.c_str(), &gdef_);
    if (!s.ok()) {
      LOG(ERROR) << "Read graph failed: " << name() << ", " << s.ToString();
      return false;
    }
  }

  graph_path_ = frozen_graph;
  size_t pos = graph_path_.find_last_of("/");
  if (pos != std::string::npos) {
    graph_path_ = graph_path_.substr(0, pos + 1);
  } else {
    graph_path_ = "./";
  }
  return true;
}

bool Model::Warmup() {
  for (auto context : predict_contexts_) {
    Session* session = context.session;
    for (int i = 0; i < inputs_.size(); i++) {
      std::vector<Tensor> outputs;
      RunMetadata meta;
      Status s = session->Run(run_options_, inputs_[i], output_names_, {}, &outputs, &meta);
      if (!s.ok()) {
        LOG(ERROR) << "Warmup: " << name() << ", Session::Run failed: " << i
                   << ", inferred_batchsize = " << inferred_batchsizes_[i] << ", " << s.ToString();
        return false;
      }
      if (outputs.size() > 0) {
        VLOG(1) << "output: " << outputs[0].DebugString();
      }
    }
  }
  return true;
}

Model* ModelReloader::CreateObject() {
  Model *model= new Model(bench_model_config_.name(), bench_model_config_.predictor_num());
  // Load graph
  std::cout << "begin load graph" << std::endl;
  if (!model->LoadGraph(bench_model_config_.frozen_graph())) {
    LOG(ERROR) << "Load graph failed: " << bench_model_config_.name();
    delete model;
    return nullptr;
  }

  std::cout << "begin parse samples" << std::endl;
  if (!model->ParseSamples(bench_model_config_.sample_file())) {
    LOG(ERROR) << "Read sample_file failed: " << bench_model_config_.sample_file() << "," 
               << bench_model_config_.name();
    delete model;
    return nullptr;
  }

  // Init TensorFlow Session
  if (!model->InitSession(bench_model_config_.config_proto())) {
    LOG(ERROR) << "Init tensorflow session failed: " << bench_model_config_.name();
    delete model;
    return nullptr;
  }

  // Prepare Session RunOptions
  if (!model->ParseRunOptions(bench_model_config_.run_options())) {
    LOG(ERROR) << "Parse run options failed: " << bench_model_config_.name()
               << ", use default RunOptions.";
  }

  // Warmup
  if (!model->Warmup()) {
    LOG(ERROR) << "Warmup failed: " << bench_model_config_.name();
    delete model;
    return nullptr;
  }

  LOG(INFO) << "Init and warmup model complete: " << bench_model_config_.name();
  return model;
}

bool ModelSelector::InitModel(
    const benchmark::BenchModelConfig& bench_model_config) {
  std::shared_ptr<ModelReloader> model_reloader =
      std::make_shared<ModelReloader>(bench_model_config);
  bool success = model_reloader->Switch();
  if (!success) {
    return false;
  }
  model_reloaders_.emplace_back(model_reloader);
  switch_interval_.emplace_back(bench_model_config.switch_interval());
  return true;
}

std::shared_ptr<Model> ModelSelector::GetModel(int idx) const {
  auto model_reloader = model_reloaders_[idx];
  return model_reloader->Instance();
}

void ModelSelector::Start() {
  running_ = true;
  std::vector<int> left_time_to_switch(switch_interval_);
  while (running_.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    for (int i = 0; i < left_time_to_switch.size(); ++i) {
      left_time_to_switch[i]--;
      if (left_time_to_switch[i] <= 0) {
        LOG(INFO) << "Begin switch model.";
        bool success = model_reloaders_[i]->Switch();
        if (!success) {
          LOG(ERROR) << "Switch model failed.";
          continue;
        }
        LOG(INFO) << "Switch model successfully.";
        left_time_to_switch[i] = switch_interval_[i];
      }
    }
  }
}

void ModelSelector::Stop() { running_ = false; }

}  // namespace benchmark
