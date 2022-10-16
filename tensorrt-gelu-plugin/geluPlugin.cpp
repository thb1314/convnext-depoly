/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <numeric>
#include <stdexcept>
#include "geluPlugin.h"
#include "gelu.h"
#include "checkMacrosPlugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::GeluPlugin;
using nvinfer1::plugin::GeluPluginCreator;

namespace
{
constexpr const char* GELU_VERSION{"1"};
constexpr const char* GELU_NAME{"Gelu"};
} // namespace

// // Static class fields initialization
PluginFieldCollection GeluPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> GeluPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GeluPluginCreator);


int GeluPlugin::initialize() noexcept
{
    return 0;
}

GeluPlugin::GeluPlugin(const void* data, size_t length)
{
    // Deserialize in the same order as serialization
}

const char* GeluPlugin::getPluginType() const noexcept
{
    return GELU_NAME;
}

const char* GeluPlugin::getPluginVersion() const noexcept
{
    return GELU_VERSION;
}

int GeluPlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DimsExprs GeluPlugin::getOutputDimensions(
    int index, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Input (from previous layer), scale and bias are the three inputs to the plugin.
    assert(nbInputs == 1);
    assert(index == 0);
    nvinfer1::DimsExprs output(inputs[0]);
    return output;
}

// Detach the plugin object from its execution context.
void GeluPlugin::detachFromContext() noexcept
{
    
}

int GeluPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // Get the input dimensions
    nvinfer1::Dims input_dims = inputDesc[0].dims;

    bool use_fp16 = inputDesc[0].type == DataType::kHALF;
    bool use_fp32 = inputDesc[0].type == DataType::kFLOAT;

    int n = std::accumulate(input_dims.d, input_dims.d + inputDesc[0].dims.nbDims, 1, std::multiplies<int>());
    
    if(use_fp16) {
        GELU::computeGelu(stream, n, (const float *)inputs[0], (float*)outputs[0]);
    } else if(use_fp32) {
        GELU::computeGelu(stream, n, (const half *)inputs[0], (half*)outputs[0]);       
    } else {
        printf("unsupported type!");
    }

    return 0;
}

size_t GeluPlugin::getSerializationSize() const noexcept
{
    return 0;
}

void GeluPlugin::serialize(void* buffer) const noexcept
{
}

bool GeluPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    assert(inOut && pos < (nbInputs + nbOutputs));
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF) && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
        && inOut[pos].type == inOut[0].type);
}

void GeluPlugin::terminate() noexcept
{

}

void GeluPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* GeluPlugin::clone() const noexcept
{
    auto* plugin = new GeluPlugin(nullptr, 0);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void GeluPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{

}

nvinfer1::DataType GeluPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

size_t GeluPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void GeluPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mPluginNamespace = libNamespace;
}

const char* GeluPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

GeluPluginCreator::GeluPluginCreator()
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GeluPluginCreator::getPluginName() const noexcept
{
    return GELU_NAME;
}

const char* GeluPluginCreator::getPluginVersion() const noexcept
{
    return GELU_VERSION;
}

const PluginFieldCollection* GeluPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

const char* GeluPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void GeluPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

IPluginV2DynamicExt* GeluPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    // Set default values
    GeluPlugin* plugin = new GeluPlugin(nullptr, 0);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}

IPluginV2DynamicExt* GeluPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    GeluPlugin* plugin = new GeluPlugin(nullptr, 0);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}
