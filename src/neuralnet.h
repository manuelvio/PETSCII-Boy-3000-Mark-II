#ifndef PB_NEURAL_NETWORK_H
#define PB_NEURAL_NETWORK_H

#include <stdint.h>

// Hyperparmeters
#define BATCH_ROW_COUNT_MAX 100
#define BATCH_ROW_LENGTH 33 // Record length = 16*2 (pixels) bytes + 1 (digit) byte
// Every digit is drawn in a 16*16 pixel box, so our input layer is made of 256 sensors
#define INPUT_LAYER_SIZE (16 * 16)
#define HIDDEN_LAYER_SIZE 14
// Every output neuron is a specific digit from 0 to 9
#define OUTPUT_LAYER_SIZE 10
#define LEARNING_RATE 0.5

#define TRAINING_RECORD_COUNT 1593
#define EPOCHS 2

// Every batch is kept in memory for performance reasons
typedef uint8_t batch_t[BATCH_ROW_COUNT_MAX][BATCH_ROW_LENGTH];

// 32 bytes representing digit pixels, every bit is a specific pixel
// We can cycle through pixels using the expression
// input[i DIV 8] AND (1 shl (7-(i mod 8)))
// where i is pixel index in loop from 0 to 255, starting from topmost left to bottom right pixel  
typedef uint8_t input_t[BATCH_ROW_LENGTH - 1];

  // Parameters

  // Hidden layer: weights, biases and activation values
float weights_hidden[INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE];
float biases_hidden[HIDDEN_LAYER_SIZE];
float activations_hidden[HIDDEN_LAYER_SIZE];

// Output layer: weights, biases and activation values
float weights_output[HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE];
float biases_output[OUTPUT_LAYER_SIZE];
float activations_output[OUTPUT_LAYER_SIZE];

float gradients_hidden[HIDDEN_LAYER_SIZE];
float gradients_output[OUTPUT_LAYER_SIZE];

// float max_output;
uint8_t batch_length;

/*
 * Sigmoid activation function
 */
float sigmoid(float x);

/*
 * Sigmoid function derivative
 */
float sigmoid_prime(float x);

/*
 * Initializes network with random values
 */
void init_network();

uint8_t predict(input_t input);

void train(input_t input, uint8_t output);

#pragma compile("neuralnet.c")

#endif