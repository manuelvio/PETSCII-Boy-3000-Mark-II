#ifndef PB_BATCH_H
#define PB_BATCH_H

#include <stdint.h>
#include "neuralnet.h"

#define BATCHES_COUNT 16

/*
 * Contains training process data
 */
typedef struct {
    batch_t batch;          // Current batch input data
    uint8_t batch_index;    // Current batch index, loop is in reverse, so when index is -1 we know that the loop has ended
    uint8_t loaded_records; // How many records have been loaded from disk
    uint16_t correct;       // Total correct guesses
    uint16_t processed;     // How many record have been processed so far

}  Training;

void init_training(Training *training);

/*
 * Loads a batch of records from disk, returns the number of loaded items
 */
void load_training_batch(uint8_t device, Training *training);

#pragma compile("batch.c")

#endif