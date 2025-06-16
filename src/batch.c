#include <stdint.h>
#include <stdio.h>
#include <c64/kernalio.h>
#include "neuralnet.h"

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

uint8_t load_training_batch(uint8_t batch_idx, uint8_t device, batch_t dest)
{
    char batch_filename[13];
    sprintf(batch_filename, "NEURAL%02X,U,R", batch_indexes[batch_idx]);
    uint8_t loaded_records = 0;
	krnio_setnam(batch_filename);	
	if (krnio_open(2, (char)device, 2)) {
        for(uint8_t row = 0; row < BATCH_ROW_COUNT_MAX; row++) {
            int ch;
            for(uint8_t row_item = 0; row_item < BATCH_ROW_LENGTH; row_item++) {
                ch = krnio_getch(2);
                if (ch & 0x100) break;
                dest[row][row_item] = ch;
            }
            if (ch & 0x100) {
                loaded_records = row + 1;
                break;
            }
        }
		krnio_close(2);
	}
    return loaded_records;
}
