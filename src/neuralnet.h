#ifndef PB_NEURAL_NETWORK_H
#define PB_NEURAL_NETWORK_H

#include <stdint.h>

// Hyperparmeters

// Every batch of training hadwritten digits can store up to 100 items
#define BATCH_ROW_COUNT_MAX 100

// Record length = 16 * 2 (pixels) bytes + 1 (digit) byte
// This is a compressed representation of initial training digit data structure:
// Every row was formed by a sequence of 256 float values (with only two values: 1.0 or 0.0) followed by ten boolean (1/0) digits
// Given that, a row can be stored in a sequence of 256 bits + a byte with value in 0-9 range
#define BATCH_ROW_LENGTH 33 
// Every digit is drawn in a 16 * 16 pixel box, so our input layer is an array of 256 sensors
#define INPUT_LAYER_SIZE (16 * 16)
// 14 hidden neurons are enough
#define HIDDEN_LAYER_SIZE 14
// Every output neuron is "mapped" to a specific digit from 0 to 9 by its position 
#define OUTPUT_LAYER_SIZE 10
#define LEARNING_RATE 0.5
// Total digits in training set
#define TRAINING_RECORD_COUNT 1593
// Two epochs are usually enough for an accuracy of 90-95%
#define EPOCHS 2

// Every batch is kept in memory for performance reasons
typedef uint8_t batch_t[BATCH_ROW_COUNT_MAX][BATCH_ROW_LENGTH];

// 32 bytes representing digit pixels, every bit is a specific pixel
// We can cycle through pixels using the expression
// input[i DIV 8] AND (1 shl (7-(i mod 8)))
// where i is pixel index in loop from 0 to 255, starting from topmost left to bottom right pixel  
typedef uint8_t input_t[BATCH_ROW_LENGTH - 1];

typedef struct {
    // Input layer values are those read from data, they basically are the pixel of the digit
    // which can either be "on" or "off": 1.0 or 0.0, there's no need to create another structure for a replica of these values

    // Hidden layer: weights, biases and activation values
    float weights_hidden[INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE]; // Every input sensor is connected to a hidden neuron, here are stored the weights of every connection
    float biases_hidden[HIDDEN_LAYER_SIZE];
    float activations_hidden[HIDDEN_LAYER_SIZE];

    // Output layer: weights, biases and activation values
    float weights_output[HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE]; // ... and every hidden neuron is connected to an output neuron, their connection weights are stored here
    float biases_output[OUTPUT_LAYER_SIZE];
    float activations_output[OUTPUT_LAYER_SIZE];

    float gradients_hidden[HIDDEN_LAYER_SIZE];
    float gradients_output[OUTPUT_LAYER_SIZE];
} NeuralNetwork;

// Neural network state
enum NeuralNetworkState {
	NS_INITIAL			// Initial state, not trained yet, weights are random 
};

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
void init_network(NeuralNetwork *neural_network);

uint8_t predict(NeuralNetwork *neural_network, input_t input);

void train(NeuralNetwork *neural_network, input_t input, uint8_t output);

#pragma compile("neuralnet.c")

#endif