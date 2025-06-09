#include <math.h>
#include <limits.h>
#include "neuralnet.h"

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

void init_network()
{
    for(unsigned int i = 0; i < (INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE); i++) {
        weights_hidden[i] = rand_float() - 0.5;
    }
    for(uint8_t h = 0; h < HIDDEN_LAYER_SIZE; h++) {
        biases_hidden[h] = 0.0;
    }
    for(uint8_t h = 0; h < (HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE); h++) {
        weights_output[h] = rand_float() - 0.5;
    }
    for(uint8_t o = 0; o < OUTPUT_LAYER_SIZE; o++) {
        biases_output[o] = 0.0;
    }
}

uint8_t predict(input_t input)
{
    float max_output = -1.0;
    uint8_t result = 0;
    
    for(uint8_t h = 0; h < HIDDEN_LAYER_SIZE; h++) {
        float sum_hidden = 0.0;
        for(uint16_t i = 0; i < INPUT_LAYER_SIZE; i++) {
            if (EXTRACT_BIT(input, i)) {
                sum_hidden += weights_hidden[i * HIDDEN_LAYER_SIZE + h];
            }
        }

        activations_hidden[h] = sigmoid(sum_hidden + biases_hidden[h]);
    }

    for(uint8_t o = 0; o < OUTPUT_LAYER_SIZE; o++) {
        float sum_output = 0.0;
        for(uint8_t h = 0; h < HIDDEN_LAYER_SIZE; h++) {
            sum_output += activations_hidden[h] * weights_output[h * OUTPUT_LAYER_SIZE + o];
        }
        activations_output[o] = sigmoid(sum_output + biases_output[o]);
        if (activations_output[o] > max_output) {
            result = o;
            max_output = activations_output[o];
        }
    }
    return result;
}

void train(input_t input, uint8_t output)
{
    uint8_t predicted = predict(input);
    
    for(uint8_t o = 0; o < OUTPUT_LAYER_SIZE; o++) {
        gradients_output[o] = (activations_output[o] - (o == output)) * sigmoid_prime(activations_output[o]);
    }

    for(uint8_t h = 0; h < HIDDEN_LAYER_SIZE; h++) {
        float gradient_hidden_sum = 0.0;
        for(uint8_t o = 0; o < OUTPUT_LAYER_SIZE; o++) {
            gradient_hidden_sum += gradients_output[o] * weights_output[h * OUTPUT_LAYER_SIZE + o];
        }
        gradients_hidden[h] = gradient_hidden_sum * sigmoid_prime(activations_hidden[h]);
    }

    for(uint8_t h = 0; h < HIDDEN_LAYER_SIZE; h++) {
        for(uint8_t o = 0; o < OUTPUT_LAYER_SIZE; o++) {
            weights_output[h * OUTPUT_LAYER_SIZE + o] -= LEARNING_RATE * gradients_output[o] * activations_hidden[h];
        }
    }

    for(uint16_t i = 0; i < INPUT_LAYER_SIZE; i++) {
        for(uint8_t h = 0; h < HIDDEN_LAYER_SIZE; h++) {
            weights_hidden[i * HIDDEN_LAYER_SIZE + h] -= LEARNING_RATE * gradients_hidden[h] * (float)EXTRACT_BIT(input, i);
        }
    }

    for(uint8_t o = 0; o < OUTPUT_LAYER_SIZE; o++) {
        biases_output[o] -= LEARNING_RATE * gradients_output[o];
    }

    for(uint8_t h = 0; h < HIDDEN_LAYER_SIZE; h++) {
        biases_hidden[h] -= LEARNING_RATE * gradients_hidden[h];
    }
}