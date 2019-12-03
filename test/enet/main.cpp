#include <iostream>
#include <memory>
#include <string>
#include <stdio.h>
#include <sstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "utils.h"

#define MODEL_FILENAME RESOURCE_DIR"enet.tflite"

#define TFLITE_MINIMAL_CHECK(x)                              \
    if (!(x)) {                                                \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }


int main()
{
    /* read input image data */
    cv::Mat image = cv::imread(RESOURCE_DIR"test.jpg", 1);
    cv::imshow("InputImage", image);

    // Initialize Colormap.
    auto color_map = CreatePascalLabelColormap();

    // Initialize Threads.
    auto num_of_threads = 4;

    cv::resize(image, image, cv::Size(480, 360));
    cv::imshow("InputImage for CNN", image);
    image.convertTo(image, CV_32FC1, 1.0 / 255);

    // TFLite
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(MODEL_FILENAME);
    TFLITE_MINIMAL_CHECK(model != nullptr);
    printf("=== Model loaded ===\n");

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);
    printf("=== Interpreter Built ===\n");

    // Get input tensor size
    const auto& dimensions = interpreter->tensor(interpreter->inputs()[0])->dims;
    auto input_array_size = 1;
    std::vector<int> input_tensor_shape;
    input_tensor_shape.resize(dimensions->size);
    for (auto i = 0; i < dimensions->size; i++)
    {
        input_tensor_shape[i] = dimensions->data[i];
        input_array_size *= input_tensor_shape[i];
    }
    std::ostringstream input_string_stream;
    std::copy(input_tensor_shape.begin(), input_tensor_shape.end(), std::ostream_iterator<int>(input_string_stream, " "));
    std::cout << "input shape: " << input_string_stream.str() << std::endl;
    std::cout << "input array size: " << input_array_size << std::endl;
    printf("=== Got model input size ===\n");

    // Allocate tensor buffers.
    interpreter->SetNumThreads(num_of_threads);
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    printf("=== Pre-invoke Interpreter State ===\n");
    //tflite::PrintInterpreterState(interpreter.get());

    // Set data to input tensor
    printf("=== MEM copy ===\n");
    float* input = interpreter->typed_input_tensor<float>(0);
    memcpy(input, image.data, sizeof(float) * input_array_size);

    // Run inference
    printf("=== Pre-invoke ===\n");
    const auto& start_time = std::chrono::steady_clock::now();
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    const std::chrono::duration<double, std::milli>& time_span = std::chrono::steady_clock::now() - start_time;
    std::cout << "Inference time: " << time_span.count() << " ms" << std::endl;
    printf("=== Post-invoke ===\n");

    // Get data from output tensor
    std::vector<float> output_data;
    const auto& output_indices = interpreter->outputs();
    const int num_outputs = output_indices.size();
    int out_idx = 0;
    for (int i = 0; i < num_outputs; ++i)
    {
        const auto* out_tensor = interpreter->tensor(output_indices[i]);
        assert(out_tensor != nullptr);
        const int num_values = out_tensor->bytes / sizeof(float);
        output_data.resize(out_idx + num_values);
        const float* output = interpreter->typed_output_tensor<float>(i);
        for (int j = 0; j < num_values; ++j)
        {
            output_data[out_idx++] = output[j];
        }
    }

    // Create segmantation map.
    cv::Mat seg_im(cv::Size(input_tensor_shape[1], input_tensor_shape[2]), CV_8UC3);
    LabelToColorMap(output_data, *color_map.get(), seg_im);

    // output tensor size => camera resolution
    cv::resize(seg_im, seg_im, cv::Size(480, 360));
    seg_im = (image / 2) + (seg_im / 2);

    cv::waitKey(0);
    return 0;
}
