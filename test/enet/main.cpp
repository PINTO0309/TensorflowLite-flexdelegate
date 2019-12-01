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
    //image = ~image;
    cv::imshow("InputImage for CNN", image);
    image.convertTo(image, CV_32FC1, 1.0 / 255);
    //cv::waitKey(0);
    //return 0;
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
    cv::waitKey(0);
    //return 0;

    // Allocate tensor buffers.
    interpreter->SetNumThreads(num_of_threads);
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    printf("=== Pre-invoke Interpreter State ===\n");
    tflite::PrintInterpreterState(interpreter.get());
    cv::waitKey(0);
    //return 0;

    // Set data to input tensor
    //float* input = interpreter->typed_input_tensor<float>(0);
    //memcpy(input, image.reshape(0, 1).data, sizeof(float) * 1 * 360 * 480 * 3);
    //memcpy(input, image.data, sizeof(float) * input_array_size);

    uint8_t* input = interpreter->typed_input_tensor<uint8_t>(0);
    std::vector<uint8_t> input_data(image.data,
                                    image.data + (image.cols * image.rows * image.elemSize()));
    memcpy(input, input_data.data(), input_data.size());

    printf("\n\n=== MEM copied ===\n");
    cv::waitKey(0);

    // Run inference
    printf("\n\n=== Pre-invoke Interpreter State ===\n");
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    printf("\n\n=== Post-invoke Interpreter State ===\n");
    tflite::PrintInterpreterState(interpreter.get());
    //cv::waitKey(0);
    //return 0;
    // Get data from output tensor
    float* probs = interpreter->typed_output_tensor<float>(0);
    for (int i = 0; i < 10; i++) {
        printf("prob of %d: %.3f\n", i, probs[i]);
    }

    cv::waitKey(0);
    return 0;
}
