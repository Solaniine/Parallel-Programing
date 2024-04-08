#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <CL/cl2.hpp>
#include <CImg.h>
#include "Utils.h"
#include <string>

using namespace cimg_library;

// Read the kernel
std::string readKernelFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return "";
    }

    std::string kernelSource;
    std::string line;
    while (getline(file, line)) {
        kernelSource += line + "\n";
    }
    file.close();

    return kernelSource;
}

//histogram equalization on 8-bit
void histogramEqualization8bit(CImg<unsigned char>& image, cl::CommandQueue& queue, cl::Context& context, cl::Device& device) {
    std::string kernelSource = readKernelFromFile("Kernels/my_kernels.cl");
    //create a program
    cl::Program::Sources sources;
    sources.push_back({ kernelSource.c_str(), kernelSource.length() });
    cl::Program program(context, sources);
    //build the program
    if (program.build({ device }) != CL_SUCCESS) {
        std::cerr << "\033[1;31mError building program:\033[0m\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return;
    }
    //creates buffers for intermediate result
    size_t bufferSize = image.size();
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, image.data());
    cl::Buffer bufferHist(context, CL_MEM_READ_WRITE, 256 * sizeof(int));
    cl::Buffer bufferCumHist(context, CL_MEM_READ_WRITE, 256 * sizeof(int));
    cl::Buffer bufferNormScaledHist(context, CL_MEM_READ_WRITE, 256 * sizeof(char));
    cl::Buffer bufferB(context, CL_MEM_WRITE_ONLY, bufferSize);

    //creates kernels for histogram cumulativeHistogram normalizeScaleHistogram backProjection
    cl::Kernel histogramKernel(program, "histogram");
    cl::Kernel cumulativeHistogramKernel(program, "cumulativeHistogram");
    cl::Kernel normalizeScaleHistogramKernel(program, "normalizeScaleHistogram");
    cl::Kernel backProjectionKernel(program, "backProjection");

    //histogram kernel
    histogramKernel.setArg(0, bufferA);
    histogramKernel.setArg(1, bufferHist);
    queue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, cl::NDRange(image.width() * image.height()), cl::NullRange);

    //cumulative histogram kernel
    cumulativeHistogramKernel.setArg(0, bufferHist);
    cumulativeHistogramKernel.setArg(1, bufferCumHist);
    queue.enqueueNDRangeKernel(cumulativeHistogramKernel, cl::NullRange, cl::NDRange(256), cl::NullRange);

    //normalization and scaling
    normalizeScaleHistogramKernel.setArg(0, bufferCumHist);
    normalizeScaleHistogramKernel.setArg(1, bufferNormScaledHist);
    queue.enqueueNDRangeKernel(normalizeScaleHistogramKernel, cl::NullRange, cl::NDRange(256), cl::NullRange);

    //back-projection kernel
    backProjectionKernel.setArg(0, bufferA);
    backProjectionKernel.setArg(1, bufferNormScaledHist);
    backProjectionKernel.setArg(2, bufferB);
    queue.enqueueNDRangeKernel(backProjectionKernel, cl::NullRange, cl::NDRange(image.width() * image.height()), cl::NullRange);

    //read the result back to host memory
    std::vector<unsigned char> outputImage(bufferSize);
    queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, bufferSize, outputImage.data());

    CImg<unsigned char> outputCImg(outputImage.data(), image.width(), image.height(), 1, image.spectrum());
    outputCImg.display("Output Image (8-bit)");
}

// 16-bit histogram equalization
void histogramEqualization16bit(CImg<unsigned short>& image, cl::CommandQueue& queue, cl::Context& context, cl::Device& device) {
 
    std::string kernelSource = readKernelFromFile("Kernels/my_kernels.cl");
    cl::Program::Sources sources;
    sources.push_back({ kernelSource.c_str(), kernelSource.length() });
    cl::Program program(context, sources);
    if (program.build({ device }) != CL_SUCCESS) {
        std::cerr << "\033[1;31mError building program:\033[0m " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return;
    }

    //creates buffers for intermediate result
    size_t bufferSize = image.size();
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize * sizeof(unsigned short), image.data());
    cl::Buffer bufferHist(context, CL_MEM_READ_WRITE, 65536 * sizeof(int));
    cl::Buffer bufferCumHist(context, CL_MEM_READ_WRITE, 65536 * sizeof(int));
    cl::Buffer bufferNormScaledHist(context, CL_MEM_READ_WRITE, 65536 * sizeof(unsigned short));
    cl::Buffer bufferB(context, CL_MEM_WRITE_ONLY, bufferSize * sizeof(unsigned short));

    //creates kernels for histogram cumulativeHistogram normalizeScaleHistogram backProjectio
    cl::Kernel histogramKernel(program, "histogram16");
    cl::Kernel cumulativeHistogramKernel(program, "cumulativeHistogram16");
    cl::Kernel normalizeScaleHistogramKernel(program, "normalizeScaleHistogram16");
    cl::Kernel backProjectionKernel(program, "backProjection16");

    //histogram kernel
    histogramKernel.setArg(0, bufferA);
    histogramKernel.setArg(1, bufferHist);
    queue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, cl::NDRange(image.width() * image.height()), cl::NullRange);

    //cumulative histogram kernel
    cumulativeHistogramKernel.setArg(0, bufferHist);
    cumulativeHistogramKernel.setArg(1, bufferCumHist);
    queue.enqueueNDRangeKernel(cumulativeHistogramKernel, cl::NullRange, cl::NDRange(65536), cl::NullRange);

    //normalization and scaling
    normalizeScaleHistogramKernel.setArg(0, bufferCumHist);
    normalizeScaleHistogramKernel.setArg(1, bufferNormScaledHist);
    queue.enqueueNDRangeKernel(normalizeScaleHistogramKernel, cl::NullRange, cl::NDRange(65536), cl::NullRange);

    //back-projection kernel
    backProjectionKernel.setArg(0, bufferA);
    backProjectionKernel.setArg(1, bufferNormScaledHist);
    backProjectionKernel.setArg(2, bufferB);
    queue.enqueueNDRangeKernel(backProjectionKernel, cl::NullRange, cl::NDRange(image.width() * image.height()), cl::NullRange);

    //read the result back to host memory
    std::vector<unsigned short> outputImage(bufferSize);
    queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, bufferSize * sizeof(unsigned short), outputImage.data());
    CImg<unsigned short> outputCImg(outputImage.data(), image.width(), image.height(), 1, image.spectrum());
    outputCImg.display("Output Image (16-bit)");
}
int main() {
    // OpenCL
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms.front();
    std::cout << "Using platform:\n " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        std::cerr << "\033[1;31mNo GPU devices found!\033[0m\n" << std::endl;
        return 1;
    }


    cl::Device device = devices.front();
    std::cout << "\nUsing device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    cl::Context context(device);
    cl::CommandQueue queue(context, device);



    bool exit = false;
    while (!exit) {
        std::string filename;
        std::cout << "Enter the filename or 'exit' to quit: ";
        std::cin >> filename;

        if (filename == "exit") {
            exit = true;
            continue;
        }
        CImg<unsigned char> image8(filename.c_str());
        CImg<unsigned short> image16(filename.c_str());

        //histogram equalization based on image depth
        if (image8.depth() == 1) {
            histogramEqualization8bit(image8, queue, context, device);
        }
        else if (image16.depth() == 2) {
            histogramEqualization16bit(image16, queue, context, device);
        }
        else {
            std::cerr << "Unsupported image depth. Please use 8-bit or 16-bit grayscale images." << std::endl;
        }
    }

    return 0;
}