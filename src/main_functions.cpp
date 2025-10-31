/* ----------------------------------------------------------------------
 * Project:  TinyTTA Engine
 *
 * Reference Paper:
 *  TinyTTA: Efficient Test-time Adaptation via Early-exit Ensembles on Edge Devices,
 *  Neural Information Processing Systems (NeurIPS) 2024
 *
 * Contact Authors:
 *  Young D. Kwon: ydk21@cam.ac.uk
 *  Hong Jia: hong.jia@unimelb.edu.au
 *  Alessio Orsino: aorsino@dimes.unical.it
 *  Ting Dang: ting.dang@unimelb.edu.au
 *  Domenico Talia: talia@dimes.unical.it
 *  Cecilia Mascolo: cm542@cam.ac.uk
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include <stdio.h>
#include <time.h>

#include "main_functions.h"

#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define DEBUG 0
// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int input_size = 1;
int output_size = 1;

int global_loop_count = 0;
constexpr int kTensorArenaSize = 5 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
int warmup_iter = 10;
int main_iter = 1000;
}  // namespace

long time_in_tick()
{
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// The name of this function is important for Arduino compatibility.
void setup() {
    tflite::InitializeTarget();
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // Common Ops
    static tflite::MicroMutableOpResolver<10> resolver;  // NOLINT
    resolver.AddReshape();
    resolver.AddAveragePool2D();
    resolver.AddPad();
    resolver.AddTranspose();
    resolver.AddMean();
    resolver.AddQuantize();
    resolver.AddAdd();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();

    // Create the allocator that will be shared between models
    tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(tensor_arena, kTensorArenaSize);
    model = tflite::GetModel(g_hello_world_int8_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("[INFO] Model provided is schema version not equal to supported version.\n\r");
        return;
    }

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(model, resolver, allocator);
    interpreter = &static_interpreter;
    printf("[INFO] AFTER: tflite::MicroInterpreter static_interpreter; !\n\r");

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return;
    }
    printf("[INFO] AFTER: TfLiteStatus allocate_status = interpreter->AllocateTensors(); !\n\r");
    input = interpreter->input(0);
    output = interpreter->output(0);
    for (int i = 0; i < input->dims->size; i++) {
        input_size *= input->dims->data[i];
    }
    for (int i = 0; i < output->dims->size; i++) {
        output_size *= output->dims->data[i];
    }
#if (defined(DEBUG) && (DEBUG == 1))
    printf("[INFO] Input Size: %d, Output Size: %d \n\r", input_size, output_size);
    float scale = input->params.scale;
    int32_t zero_point = input->params.zero_point;
    printf("[INFO] Input Param (Scale, Zero Point): (%f, %d) \n\r", scale, zero_point);
    scale = output->params.scale;
    zero_point = output->params.zero_point;
    printf("[INFO] Output Param (Scale, Zero Point): (%f, %d) \n\r", scale, zero_point);
#endif
    printf("[INFO] End of Setup() !\n\r");
}

void loop() {
    long t_infer_head = 0;
    long t_infer_head_start = 0;

    long t_infer_total = 0;
    long t_infer_total_start = 0;

    //////////////////////////////////////////////////
    /////// EXECUTION PART BELOW /////////////////////
    printf("[INFO] EXECUTION STARTS !\n\r");
    global_loop_count += 1;

    // Set up dummy inputs
    std::default_random_engine rand_generator;
    std::normal_distribution<float> dist(3, 2);

    float input_fp;
    float output_fp;
    int8_t* input_buffer = tflite::GetTensorData<int8_t>(interpreter->input(0));
    int8_t* output_buffer = tflite::GetTensorData<int8_t>(interpreter->output(0));
    for (int i = 0; i < input_size; i++) {
        input_buffer[i] = dist(rand_generator) / input->params.scale + input->params.zero_point;
    }

    // warm up 10 iterations
    printf("[INFO] WARM-UP STARTS !\n\r");
    for (int i = 0; i < warmup_iter; i++) {
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed \n\r");
            return;
        }
    }

    t_infer_total_start = time_in_tick();
    printf("[INFO] MAIN LOOP STARTS !\n\r");
    // measure 100 iterations
    for (int i = 0; i < main_iter; i++) {

        ///// MODEL /////
        t_infer_head_start = time_in_tick();
        for (int k = 0; k < input_size; k++) {
            input_fp = dist(rand_generator);
            input_buffer[k] = input_fp / input->params.scale + input->params.zero_point;
        }
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed \n\r");
            printf("[INFO] MAIN LOOP: Iteration count: %d, HEAD MODEL: Invoke failed !\n\r", i);
            return;
        }
        t_infer_head += (time_in_tick() - t_infer_head_start);

        ///// RESULT PROCESSING /////
        output_fp = (output_buffer[0] - output->params.zero_point) * output->params.scale;
    }
    t_infer_total = (time_in_tick() - t_infer_total_start);

    printf("\n\rGLOBAL LOOP Count: %d, Main Iteration Count: %d\n\r", global_loop_count, main_iter);
    printf("Model Execution Time  : %f milliseconds\n\r", ((double)t_infer_head / main_iter));
    printf("Total Latency         : %f milliseconds\n\r",  ((double)t_infer_total));
    /////// EXECUTION PART ABOVE ////////
}
