#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "cppflow/cppflow.h"
#include <tensorflow/c/c_api.h>

namespace tf = tensorflow;

int test() {
    // Set the logging verbosity to error
    //_putenv("TF_CPP_MIN_LOG_LEVEL=2");

    // Ouverture du modèle SavedModel
    const char* model_path = "models/regular";
    const char* tags = "serve";
    TF_SessionOptions* session_options = TF_NewSessionOptions();
    TF_Buffer* run_options = nullptr;
    TF_Status* status = TF_NewStatus();

    TF_Graph* graph = TF_NewGraph();
    TF_Session* session = TF_LoadSessionFromSavedModel(session_options, run_options, model_path, &tags, 1, graph, nullptr, status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Error loading SavedModel: " << TF_Message(status) << std::endl;
        TF_DeleteSessionOptions(session_options);
        TF_DeleteGraph(graph);
        TF_DeleteStatus(status);
        return 1;
    }

     /*int n_ops = 5000;
     for (int i = 0; i < n_ops; i++)
     {
         size_t pos = i;
         std::cout << "Input: " << TF_OperationName(TF_GraphNextOperation(graph, &pos)) << "\n";
     }*/

    // Récupération du nom des entrées et des sorties
    const int num_inputs = 1; // TF_GraphNumInputs(graph);
    const int num_outputs = 1; // TF_GraphNumOutputs(graph);
    std::vector<TF_Output> inputs(num_inputs);
    std::vector<TF_Output> outputs(num_outputs);

    // Récupération des noms des opérations d'entrée et de sortie du modèle
    // Ces noms sont spécifiques au modèle et peuvent être récupérés à partir du graphe avec TensorBoard par exemple
    const char* input_op_name = "serving_default_input_0";
    const char* output_op_name = "StatefulPartitionedCall"; // "StatefulPartitionedCall";

    // Récupération des index des opérations d'entrée et de sortie du modèle
    inputs[0] = { TF_GraphOperationByName(graph, input_op_name), 0 };
    if (inputs[0].oper == nullptr) {
        std::cerr << "Cannot find input operation: " << input_op_name << std::endl;
        TF_CloseSession(session, status);
        TF_DeleteSession(session, status);
        TF_DeleteSessionOptions(session_options);
        TF_DeleteGraph(graph);
        TF_DeleteStatus(status);
        return 1;
    }
    outputs[0] = { TF_GraphOperationByName(graph, output_op_name), 0 };
    if (outputs[0].oper == nullptr) {
        std::cerr << "Cannot find output operation: " << output_op_name << std::endl;
        TF_CloseSession(session, status);
        TF_DeleteSession(session, status);
        TF_DeleteSessionOptions(session_options);
        TF_DeleteGraph(graph);
        TF_DeleteStatus(status);
        return 1;
    }

    // Préparation des données d'entrée
    cv::Mat input_image = cv::imread("input.png", cv::IMREAD_COLOR);
    if (input_image.empty()) {
        std::cerr << "Cannot load input image." << std::endl;
        TF_CloseSession(session, status);
        TF_DeleteSession(session, status);
        TF_DeleteSessionOptions(session_options);
        TF_DeleteGraph(graph);
        TF_DeleteStatus(status);
        return 1;
    }

    // Dimension du tensor d'entrée
    const int64_t dims[] = { 1, input_image.rows, input_image.cols, input_image.channels() };

    // Création du tensor d'entrée
    const int data_size = input_image.rows * input_image.cols * input_image.channels() * sizeof(float);
    const auto data = input_image.ptr<float>();
    const auto input_tensor = TF_NewTensor(TF_FLOAT, dims, 4, data, data_size, [](void* data, size_t, void*) { free(data); }, nullptr);

    // Exécution du modèle
    std::vector<TF_Tensor*> output_tensors(num_outputs);
    TF_SessionRun(session, nullptr, inputs.data(), &input_tensor, 1, outputs.data(), output_tensors.data(), num_outputs, nullptr, 0, nullptr, status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Error running model: " << TF_Message(status) << std::endl;
        TF_CloseSession(session, status);
        TF_DeleteSession(session, status);
        TF_DeleteSessionOptions(session_options);
        TF_DeleteGraph(graph);
        TF_DeleteStatus(status);
        return 1;
    }

    // Récupération de l'image de sortie
    const auto output_tensor = output_tensors[0];
    const auto output_data = static_cast<float*>(TF_TensorData(output_tensor));
    const int output_height = TF_Dim(output_tensor, 1);
    const int output_width = TF_Dim(output_tensor, 2);
    const int output_channels = TF_Dim(output_tensor, 3);
    cv::Mat output_image(output_height, output_width, CV_32FC(output_channels), output_data);
    
    // Conversion de l'image de sortie en format OpenCV
    cv::Mat output_image_8u;
    output_image.convertTo(output_image_8u, CV_8UC(output_channels));
    
    // Enregistrement de l'image de sortie
    const char* output_path = "output.png";
    cv::imwrite(output_path, output_image_8u);
    
    // Libération des ressources
    TF_CloseSession(session, status);
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(session_options);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
    
    return 0;
}

cv::Mat normalizeImage(const cv::Mat& input) {
    cv::Mat output;
    cv::normalize(input, output, 0, 255, cv::NORM_MINMAX, CV_8UC3);
    return output;
}

int main() {
     //return test(); // test using Tensorflow for C (crashing when running the model)

    // Read the graph
    cppflow::model model("models/regular");
    //cppflow::model model("models/realesrgan-x4");

    // Load an image
    auto input = cppflow::decode_jpeg(cppflow::read_file(std::string("test.png"))); // 640x444

    // Cast it to float, normalize to range [0, 1], and add batch_dimension
    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    //input = input / 255.f;
    input = cppflow::expand_dims(input, 0);

    // From OpenCV (test)
    //cv::Mat image = cv::imread("input.jpg");
    //cppflow::tensor input_tensor = cppflow::tensor(image);
    //input_tensor = input_tensor / 255.0f;
    //input_tensor = cppflow::expand_dims(input_tensor, 0);

    // Run
    try {
        //auto output = model({ {"serving_default_input.1:0", input} }, { "PartitionedCall:0" }); // realesrgan-x4
        auto output = model({ {"serving_default_input_0:0", input} }, { "StatefulPartitionedCall:0" }); // regular
        //auto output = model({ {"serving_default_input_0:0", input_tensor} }, { "StatefulPartitionedCall:0" });


        // Show the predicted class
        std::cout << cppflow::arg_max(output[0], 1) << std::endl;

        // Save it into an image    
        auto output_tensor = output[0];
        output_tensor = cppflow::cast(output_tensor, TF_FLOAT, TF_UINT8);

        std::vector<float> output_data = output[0].get_data<float>();
        // int rows = 444 * 4; // sonic
        // int cols = 640 * 4;
        int rows = 120 * 4; // ffvii
        int cols = 125 * 4;
        std::cout << output_data.size() << std::endl;

        cv::Mat output_image_rgb(rows, cols, CV_32FC3, output_data.data());
        cv::Mat output_image_bgr;
        cv::cvtColor(output_image_rgb, output_image_bgr, cv::COLOR_RGB2BGR);
        //output_image_bgr = output_image_bgr * 255.0f;
        output_image_bgr.convertTo(output_image_bgr, CV_8UC3);
        //convertScaleAbs(output_image_bgr, output_image_bgr);
        //cv::Mat output_image = normalizeImage(output_image_bgr);
        cv::imwrite("output_test.png", output_image_bgr);
    }
    catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
