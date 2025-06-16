#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include "neuralnet.h"

/*
MIT License

Copyright (c) 2025-Present Manuel Vio

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Extract specific bit value from an array of 32 bytes, index can range from 0 to 255
// Here we treat the byte array as a bit stream, or a linear representation of an input matrix
// where dimensions are 16x16 boolean values
#define EXTRACT_BIT(arr, i) ((arr)[(i) >> 3] >> (7 - ((i) & 7)) & 1)

float rand_float()
{
    return (float)rand() / UINT_MAX;
}

float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x)); 
}

float sigmoid_prime(float x)
{
    return x * (1.0 - x);
}

void init_network(NeuralNetwork *neural_network)
{
    for(unsigned int i = 0; i < (INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE); i++) {
        neural_network->weights_hidden[i] = rand_float() - 0.5;
    }
    for(uint8_t h = 0; h < HIDDEN_LAYER_SIZE; h++) {
        neural_network->biases_hidden[h] = 0.0;
    }
    for(uint8_t h = 0; h < (HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE); h++) {
        neural_network->weights_output[h] = rand_float() - 0.5;
    }
    for(uint8_t o = 0; o < OUTPUT_LAYER_SIZE; o++) {
        neural_network->biases_output[o] = 0.0;
    }
}

uint8_t predict(NeuralNetwork *neural_network, input_t input)
{
    float max_output = -1.0;
    uint8_t result = 0;
    
    for(uint8_t h = 0; h < HIDDEN_LAYER_SIZE; h++) {
        float sum_hidden = 0.0;
        for(uint16_t i = 0; i < INPUT_LAYER_SIZE; i++) {
            if (EXTRACT_BIT(input, i)) {
                sum_hidden += neural_network->weights_hidden[i * HIDDEN_LAYER_SIZE + h];
            }
        }

        neural_network->activations_hidden[h] = sigmoid(sum_hidden + neural_network->biases_hidden[h]);
    }

    for(uint8_t o = 0; o < OUTPUT_LAYER_SIZE; o++) {
        float sum_output = 0.0;
        for(uint8_t h = 0; h < HIDDEN_LAYER_SIZE; h++) {
            sum_output += neural_network->activations_hidden[h] * neural_network->weights_output[h * OUTPUT_LAYER_SIZE + o];
        }
        neural_network->activations_output[o] = sigmoid(sum_output + neural_network->biases_output[o]);
        if (neural_network->activations_output[o] > max_output) {
            result = o;
            max_output = neural_network->activations_output[o];
        }
    }
    return result;
}

void train(NeuralNetwork *neural_network, input_t input, uint8_t output)
{
    uint8_t predicted = predict(neural_network, input);
    
    for(uint8_t o = 0; o < OUTPUT_LAYER_SIZE; o++) {
        neural_network->gradients_output[o] = (neural_network->activations_output[o] - (o == output)) * sigmoid_prime(neural_network->activations_output[o]);
    }

    for(uint8_t h = 0; h < HIDDEN_LAYER_SIZE; h++) {
        float gradient_hidden_sum = 0.0;
        for(uint8_t o = 0; o < OUTPUT_LAYER_SIZE; o++) {
            gradient_hidden_sum += neural_network->gradients_output[o] * neural_network->weights_output[h * OUTPUT_LAYER_SIZE + o];
        }
        neural_network->gradients_hidden[h] = gradient_hidden_sum * sigmoid_prime(neural_network->activations_hidden[h]);
    }

    for(uint8_t h = 0; h < HIDDEN_LAYER_SIZE; h++) {
        for(uint8_t o = 0; o < OUTPUT_LAYER_SIZE; o++) {
            neural_network->weights_output[h * OUTPUT_LAYER_SIZE + o] -= LEARNING_RATE * neural_network->gradients_output[o] * neural_network->activations_hidden[h];
        }
    }

    for(uint16_t i = 0; i < INPUT_LAYER_SIZE; i++) {
        for(uint8_t h = 0; h < HIDDEN_LAYER_SIZE; h++) {
            neural_network->weights_hidden[i * HIDDEN_LAYER_SIZE + h] -= LEARNING_RATE * neural_network->gradients_hidden[h] * (float)EXTRACT_BIT(input, i);
        }
    }

    for(uint8_t o = 0; o < OUTPUT_LAYER_SIZE; o++) {
        neural_network->biases_output[o] -= LEARNING_RATE * neural_network->gradients_output[o];
    }

    for(uint8_t h = 0; h < HIDDEN_LAYER_SIZE; h++) {
        neural_network->biases_hidden[h] -= LEARNING_RATE * neural_network->gradients_hidden[h];
    }
}