#include <stdint.h>
#include <stdio.h>
#include <c64/kernalio.h>
#include "neuralnet.h"
#include "batch.h"

/*
 * This array is used to determine batches loading order
 * We could use a feistel network to obtain a random sequence instead,
 * but given the very limited number of items an array with a shuffling
 * function is still a better solution in terms of space and simplicity.
 * We're going to store here all batches indexes multiplied for the epochs,
 * then we'll simply obtain the next index and load the corresponding
 * batch from disk.
 */
uint8_t batch_indexes[EPOCHS * BATCHES_COUNT];

/*
 * Shuffles an array of bytes using Sattolo's algorithm
 * Sandra Sattolo was my Computer Science teacher back in the high school days... grazie prof!
 */
void shuffle_array(uint8_t arr[], int size)
{
    int i = size;
    while(i > 1) {
        int j = rand() % --i;
        uint8_t temp = arr[j];
        arr[j] = arr[i];
        arr[i] = temp;
    }
}

/*
 * Initializes training data structure
 */
void init_training(Training *training)
{
    // We will the array with indexes, repeated EPOCHS times,
    // and shuffle them in their epoch section.
    // Every epoch should process a batch only once, so the sequence obtained
    // at the end should reflect this constraint.
    for(uint8_t e = 0; e < EPOCHS; e++) {
        for(uint8_t i = 0; i < BATCHES_COUNT; i++) {
            batch_indexes[i + (e * BATCHES_COUNT)] = i;
        }

        // Passing the first element address and the number of elements to shuffle
        shuffle_array(&batch_indexes[e * BATCHES_COUNT], BATCHES_COUNT);
    }
    training->batch_index = (EPOCHS * BATCHES_COUNT) - 1;
    training->processed = 0;
    training->correct = 0;
}

void load_training_batch(uint8_t device, Training *training)
{
    char batch_filename[13];
    sprintf(batch_filename, "NEURAL%02X,U,R", batch_indexes[training->batch_index]);
    training->loaded_records = 0;
	krnio_setnam(batch_filename);
	if (krnio_open(2, (char)device, 2)) {
        for(uint8_t row = 0; row < BATCH_ROW_COUNT_MAX; row++) {
            int ch;
            for(uint8_t row_item = 0; row_item < BATCH_ROW_LENGTH; row_item++) {
                ch = krnio_getch(2);
                if (ch & 0x100) break;
                training->batch[row][row_item] = ch;
            }
            if (ch & 0x100) {
                training->loaded_records = row + 1;
                break;
            }
        }
		krnio_close(2);
	}
}
