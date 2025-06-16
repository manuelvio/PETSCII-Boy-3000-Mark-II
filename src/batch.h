#ifndef PB_BATCH_H
#define PB_BATCH_H

#include <stdint.h>
#include "neuralnet.h"



uint8_t batch_indexes[] = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F };

/*
 * Shuffles an array of bytes using Sattolo's algorithm
 * Lovely coincidence, Sandra Sattolo was my Computer Science teacher back in the high school days... grazie prof!
 */
void shuffle_array(uint8_t arr[], int size);

/*
 * Loads a batch of records from disk, returns the number of loaded items
 */
uint8_t load_training_batch(uint8_t batch_idx, uint8_t device, batch_t dest);

#pragma compile("batch.c")

#endif