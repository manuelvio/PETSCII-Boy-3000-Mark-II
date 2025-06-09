#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <conio.h>
#include <oscar.h>
#include <c64/joystick.h>
#include <c64/memmap.h>
#include <c64/vic.h>
#include <c64/charwin.h>
#include <c64/keyboard.h>
#include <c64/kernalio.h>
#include <c64/sprites.h>
#include <c64/cia.h>
#include "neuralnet.h"
#include "stringutils.h"
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

#pragma region(main, 0x0a00, 0xcc00, , , {code, data, bss, heap, stack})

// Load custom charset at 0xc000, we'll copy it later where needed (0xd000) and reuse this memory area as sprite data
#pragma section(charset, 0)
#pragma region(charset, 0xc000, 0xc800, , , {charset})
#pragma data(charset)
char charset[2048] = {
	#embed 2048 "../resources/petsciiboy-charset.bin"
};

/* Here is loaded background screen data in Petmate prg format:
 * 2 bytes: destination address
 * 1 byte: border color
 * 1 byte: background color
 * 1000 bytes: screencodes
 * 1000 bytes: color data
 * We can skip the first 4 bytes since address, background and border colors are going to be set in code */
#pragma section(petmate, 0)
#pragma region(petmate, 0xb800, 0xc000, , , {petmate})
#pragma data(petmate)

char petmate_screen[1000] = {
	#embed 1000 4 "../resources/background.bin"
};
char petmate_color[1000] = {
	#embed 1000 1004 "../resources/background.bin"
};

#pragma section(sprites, 0)
#pragma region(sprites, 0xb770, 0xb800, , , {sprites})
#pragma data(sprites)

const char sprite_data[] = {
	#embed spd_sprites lzo "../resources/neural_sprites.spd"
};


#pragma data(data)

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define CLAMP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

#define R6510   ((char *)0x0001)
#define Screen	((char *)0xcc00)
#define Charset	((char *)0xd000)
#define Color	((char *)0xd800)
#define Sprite0 ((char *)0xc000)
#define LightPenX ((char *)0xd013)
#define LightPenY ((char *)0xd014)

#define DRIVE_NO 8

// Terminal window dimensions in screen characters
#define TERMINAL_TOP    19
#define TERMINAL_LEFT   1
#define TERMINAL_HEIGHT 4
#define TERMINAL_WIDTH  37

#define TERMINAL_COLOR  VCOL_GREEN

// Menu window dimensions in screen characters
#define MENU_TOP    6
#define MENU_LEFT   19
#define MENU_HEIGHT 11
#define MENU_WIDTH  20

#define MENU_COLOR  VCOL_GREEN

// Canvas window dimensions in screen characters
#define CANVAS_TOP    1
#define CANVAS_LEFT   1
#define CANVAS_HEIGHT 16
#define CANVAS_WIDTH  16

#define CANVAS_COLOR  VCOL_GREEN

#define CANVAS_PIXEL_OFF ' '
#define CANVAS_PIXEL_ON '*'

#define BATCHES_COUNT 16


CharWin cw_menu;
CharWin cw_terminal;
CharWin cw_canvas;

char terminal_buf[37];
char batch_filename[13];
batch_t batch;
uint8_t batch_indexes[] = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F };
static const char activation_histogram_levels[9] = { 32, 100, 111, 121, 98, 248, 247, 227, 224 };


// Game state
enum ApplicationState {
	AS_READY,			// Getting ready
	AS_TRAINING,		// Network is training
	AS_DRAWING,		    // User is drawing a digit
	AS_LOADING,		    // Loading parameters
	AS_SAVING		    // Saving parameters
};

// Available input methods for handwritten digits
enum InputMode {
    JOYSTICK,
    LIGHT_PEN
} input_mode = LIGHT_PEN;

// Current state of the application
struct Application {
	ApplicationState	state;		// State
    uint8_t             epochs_left;     // Epochs to be processed
    uint8_t             batches_left;
}	TheApplication;

static const char * main_menu_texts[] = {
  "F1-TRAIN",
  "F3-SAVE PARAMS",
  "F5-LOAD PARAMS",
  "F7-DRAW DIGIT"
};

static const char * training_menu_texts[] = {
  "PRESS ANY KEY",
  "TO STOP"
};

static const char * drawing_menu_texts[] = {
  "F1-JOYSTICK",
  "F3-LIGHT PEN",
  "F5-PREDICT"
};

// Small round function to be inlined
static inline int8_t round_to_uint8(float x) {
    return (x >= 0) ? (int8_t)(x + 0.5) : (int8_t)(x - 0.5);
}

// Forward declaration
void application_state(ApplicationState state);

/*
 * Shuffles an array of bytes using Sattolo's algorithm
 * Lovely coincidence, Sandra Sattolo was my Computer Science teacher back in the high school days... grazie prof!
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
 * Displays main menu data
 */
void display_menu(const char **menu, uint8_t items_count)
{
    cwin_fill_rect(&cw_menu, 0, 0, cw_menu.wx, cw_menu.wy, ' ', MENU_COLOR);
    for (uint8_t entry_id = 0; entry_id < items_count; entry_id++) {
        cwin_putat_string(&cw_menu, 0, entry_id, menu[entry_id], MENU_COLOR);
    }
}

/*
 * Wraps kernal calls to save a memory area into a file
 */
int save_bytes(const char *filename, uint8_t device, void *source, int size)
{
    int result = -1;
	krnio_setnam(filename);	
	if (krnio_open(2, (char)device, 2)) {
		result = krnio_write(2, (char *)source, size);
		krnio_close(2);
	}
    return result;
}

/*
 * Wraps kernal calls to load a file into a memory area
 */
int load_bytes(const char *filename, uint8_t device, void *dest, int size)
{
    int result = -1;
	krnio_setnam(filename);	
	if (krnio_open(2, (char)device, 2)) {
		result = krnio_read(2, (char *)dest, size);
		krnio_close(2);
	}
    return result;
}

/*
 * Scrolls up terminal and adds a new row of text
 */
void terminal_log(const char *str)
{
    cwin_scroll_up(&cw_terminal, 1);
    cwin_fill_rect(&cw_terminal, 0, cw_terminal.wy-1, cw_terminal.wx, 1, ' ', TERMINAL_COLOR);
    cwin_putat_string(&cw_terminal, 0, cw_terminal.wy-1, str, TERMINAL_COLOR);
}


/*
 * Loads a batch of records from disk, returns the number of loaded items
 */
uint8_t load_training_batch(const char *filename, uint8_t device, batch_t dest)
{
    uint8_t loaded_records = 0;
	krnio_setnam(filename);	
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

/*
 * Display a message and wait for a Yes/No answer, returns true/false accordingly
 * In this case everything different from a 'Y' is false
 */
bool confirm(const char * msg)
{
    terminal_log(msg);
    return getch() == 89;
}

/*
 * Copy an array of bytes representing a digit into sprite 0 memory
 * WARNING: this function is incompatible with O2 optimization, somehow dx value
 * in the lvalue is different from dx in the rvalue. This issue does not exist when compiled
 * with minor optimization levels
 */
void draw_digit(input_t input)
{
    for(uint8_t dy = 0; dy < 16; dy++) {
        for(uint8_t dx = 0; dx < 2; dx++) {
            Sprite0[(dy * 3) + dx] = input[(dy * 2) + dx];
        }
    }
}

/*
 * Draws a histogram using PETSCII characters starting at a specific screen coordinate
 * x and y arguments refer to screen coordinates
 */
void petscii_histogram(uint8_t x, uint8_t y, float values[], uint8_t num_values)
{
    for(uint8_t i = 0; i < num_values; i++) {
        Screen[y * 40 + x + i] = activation_histogram_levels[round_to_uint8(values[i] * 8)];
    }
}


/*
 * Main training loop, iterates over the input batches for the epochs number
 * Every batch is used for training and checked to compute overall network accuracy
 */
void train_loop(uint8_t epochs)
{
    bool stop = false;
    uint8_t loaded = 0;
    uint16_t total = epochs * TRAINING_RECORD_COUNT;
    terminal_log("IT WILL TAKE A LOT OF TIME");
    terminal_log("MAKE A CUP OF TEA");
    terminal_log("PUT A RECORD ON");
    init_network();
    for(uint8_t epoch = 0; epoch < epochs; epoch++) {
        if (stop) break;

        uint16_t correct = 0;
        uint16_t processed = 0;

        // Batches should be loaded in random order, so we're shuffling their indexes
        shuffle_array(batch_indexes, ARRAY_SIZE(batch_indexes));

        for(uint8_t b = 0; b < ARRAY_SIZE(batch_indexes); b++) {
            if (stop) break;

            // Load batch data in memory
            sprintf(batch_filename, "NEURAL%02X,U,R", batch_indexes[b]);
            sprintf(terminal_buf, "LOADING %d - %d NEURAL%02X", b, batch_indexes[b], batch_indexes[b]);
            terminal_log(terminal_buf);
            spr_show(0, false);
            batch_length = load_training_batch(batch_filename, DRIVE_NO, batch);
            spr_show(0, true);
            
            // Training phase
            for(uint8_t record = 0; record < batch_length; record++) {
                if (stop) break;
                draw_digit(batch[record]);
                train(batch[record], batch[record][BATCH_ROW_LENGTH - 1]);
                sprintf(terminal_buf, "RECORDS REMAINING: %d", --total);
                terminal_log(terminal_buf);
                if (kbhit()) {
                    stop = confirm("STOP TRAINING? (Y/N)");
                }
            }

            // Verify prediction on batch data after training
            for(uint8_t record = 0; record < batch_length; record++) {
                if (stop) break;
                processed++;
                draw_digit(batch[record]);
                uint8_t predicted  = predict(batch[record]);
                petscii_histogram(19, 2, activations_hidden, HIDDEN_LAYER_SIZE);
                petscii_histogram(19, 4, activations_output, OUTPUT_LAYER_SIZE);
                correct += batch[record][BATCH_ROW_LENGTH - 1] == predicted;
                sprintf(terminal_buf, "ACCURACY=%.2f", ((float)correct / processed) * 100);
                terminal_log(terminal_buf);
                if (kbhit()) {
                    stop = confirm("STOP TRAINING? (Y/N)");
                }
            }
        }
    }
    if (stop) {
        terminal_log("TRAINING STOPPED");
    }
    spr_show(0, false);
}

/*
 * Converts canvas handwritten data to a byte array
 */
void canvas_to_input(CharWin * win, input_t input)
{
    uint8_t byte_value = 0;
    for(uint8_t y = 0; y < win->wy; y++) {
        for(uint8_t x = 0; x < win->wx; x++) {
            uint8_t i = (y * win->wx) + x;
            if (cwin_getat_char(win, x, y) == CANVAS_PIXEL_ON) {
                byte_value |= 1 << (7 - (i & 7));
            }
            if ((i & 7) == 7) {
                input[i >> 3] = byte_value;
                byte_value = 0;
            }            
        }
    }
}

/*
 * Displays in the canvas a digit taken from ROM charset 
 */
void display_char(uint8_t digit)
{
    uint8_t chardata[8]; // The glyph data (8 bytes) is copied here
    cwin_fill_rect(&cw_canvas, 0, 0, cw_canvas.wx, cw_canvas.wy, ' ', CANVAS_COLOR);
    cia1.cra &= 0xFE; // Disable interrupt
    *R6510 &= 0xFB; // Enable Charset rom
    for(uint8_t row = 0; row < 8; row++) {
        chardata[row] = Charset[(48 + digit) * 8 + row];
    }
    *R6510 |= 0x04; // Disable charset rom
    cia1.cra |= 0x01; // Re-enable interrupt
    for(uint8_t row = 0; row < 8; row++) {
        for(uint8_t bit = 0; bit < 8; bit++) {
            if(chardata[row] & (1 << (7 - bit))) {
                // Every original pixel is "doubled" horizontally and vertically to better fit into canvas
                cwin_putat_char(&cw_canvas, bit << 1, row << 1, CANVAS_PIXEL_ON, VCOL_GREEN);
                cwin_putat_char(&cw_canvas, bit << 1, (row << 1) + 1, CANVAS_PIXEL_ON, VCOL_GREEN);
                cwin_putat_char(&cw_canvas, (bit << 1) + 1, row << 1, CANVAS_PIXEL_ON, VCOL_GREEN);
                cwin_putat_char(&cw_canvas, (bit << 1) + 1, (row << 1) + 1, CANVAS_PIXEL_ON, VCOL_GREEN);
            }
        }
    }
}


/*
 * Let the user write a digit and, once finished, tries to recognize it
 * After the prediction user's feedback is used to further finetune network parameters
 */
void draw_and_predict()
{
    input_t current_input;
    cwin_fill_rect(&cw_canvas, 0, 0, cw_canvas.wx, cw_canvas.wy, ' ', CANVAS_COLOR);
    bool done = false;
    spr_show(1, true);
    do {
        bool moved = false;
        bool toggled = false;
        if (input_mode == JOYSTICK) {
            joy_poll(0);
            cw_canvas.cx = CLAMP((signed char)cw_canvas.cx + joyx[0], 0, cw_canvas.wx - 1);
            cw_canvas.cy = CLAMP((signed char)cw_canvas.cy + joyy[0], 0, cw_canvas.wy - 1);
            moved = true;
            toggled = (bool)joyb[0];
        } else if (input_mode == LIGHT_PEN) {
            // Read light pen position, convert it to screen coordinates and then to canvas coordinates
            // X position has to be multiplied by two, since its value was halved in order to be stored in a byte
            uint8_t light_pen_cx = (((*LightPenX << 1) - 48) >> 3) - cw_canvas.sx;
            uint8_t light_pen_cy = ((*LightPenY - 50) >> 3) - cw_canvas.sy;
            if (light_pen_cx != cw_canvas.cx && light_pen_cx >= 0 && light_pen_cx < cw_canvas.wx) {
                cw_canvas.cx = light_pen_cx;
                moved = true;                
            }
            if (light_pen_cy != cw_canvas.cy && light_pen_cy >= 0 && light_pen_cy < cw_canvas.wy) {
                cw_canvas.cy = light_pen_cy;
                moved = true;
            }
            toggled = moved;         
        }
        if (moved) {
            spr_move(1, ((cw_canvas.cx + cw_canvas.sx) << 3) + 24, ((cw_canvas.cy + cw_canvas.sy) << 3) + 50);            
            if (toggled) {
                char prev_char = cwin_getat_char(&cw_canvas, cw_canvas.cx, cw_canvas.cy);
                cwin_putat_char(&cw_canvas, cw_canvas.cx, cw_canvas.cy, prev_char == CANVAS_PIXEL_OFF ? CANVAS_PIXEL_ON : CANVAS_PIXEL_OFF, VCOL_GREEN);
            }
        }
        if (kbhit()) {
            char c = getch();
            switch (c) {
            case PETSCII_F1:
                input_mode = JOYSTICK;
                terminal_log("INPUT MODE: JOYSTICK");
                break;
            case PETSCII_F3:
                input_mode = LIGHT_PEN;
                terminal_log("INPUT MODE: LIGHTPEN");
                break;
            case PETSCII_F5:
                done = confirm("EXIT DRAWING? (Y/N)");
                break;
            }
        }
        // Loop is too fast, wait a few frames
        vic_waitFrames(5);
    } while (!done);
    spr_show(1, false);

    // Prediction and result display
    canvas_to_input(&cw_canvas, current_input);
    draw_digit(current_input);
    spr_show(0, true);
    uint8_t predicted = predict(current_input);
    petscii_histogram(19, 2, activations_hidden, HIDDEN_LAYER_SIZE);
    petscii_histogram(19, 4, activations_output, OUTPUT_LAYER_SIZE);
    display_char(predicted);
    sprintf(terminal_buf, "I THINK YOU WROTE A %d", predicted);
    terminal_log(terminal_buf);

    // Collecting user's feedback about prediction and adjusting parameters
    if (!confirm("AM I RIGHT?")) {
        bool is_number = false;
        do {
            terminal_log("ENTER THE CORRECT DIGIT (0-9)");
            char answer = getch();
            is_number = answer >= 48 && answer <= 57;
            if (is_number) predicted = answer - 48;
        } while (!is_number);
    }
    terminal_log("ADJUSTING WEIGHTS...");
    train(current_input, predicted);
    spr_show(0, false);
}

/*
 * Application state based flow
 */
void application_state(ApplicationState state)
{
	// Set new state
	TheApplication.state = state;

	switch (state) {
	case AS_READY:
        display_menu(main_menu_texts, ARRAY_SIZE(main_menu_texts));
        break;
	case AS_TRAINING:
        display_menu(training_menu_texts, ARRAY_SIZE(training_menu_texts));
        train_loop(EPOCHS);
        application_state(AS_READY);
        break;
	case AS_SAVING:
        cwin_fill_rect(&cw_menu, 0, 0, cw_menu.wx, cw_menu.wy, ' ', MENU_COLOR);
        if (confirm("ARE YOU SURE? (Y/N)")) {
            terminal_log("SAVING PARAMETERS...");
            save_bytes("@0:WH,U,W", DRIVE_NO, weights_hidden, sizeof(weights_hidden));
            save_bytes("@0:WO,U,W", DRIVE_NO, weights_output, sizeof(weights_output));
            save_bytes("@0:BH,U,W", DRIVE_NO, biases_hidden, sizeof(biases_hidden));
            save_bytes("@0:BO,U,W", DRIVE_NO, biases_output, sizeof(biases_output));
            terminal_log("...DONE");
        }
        application_state(AS_READY);
        break;
	case AS_LOADING:
        cwin_fill_rect(&cw_menu, 0, 0, cw_menu.wx, cw_menu.wy, ' ', MENU_COLOR);
        if (confirm("ARE YOU SURE? (Y/N)")) {
            terminal_log("LOADING PARAMETERS...");
            load_bytes("WH,U,R", DRIVE_NO, weights_hidden, sizeof(weights_hidden));
            load_bytes("WO,U,R", DRIVE_NO, weights_output, sizeof(weights_output));
            load_bytes("BH,U,R", DRIVE_NO, biases_hidden, sizeof(biases_hidden));
            load_bytes("BO,U,R", DRIVE_NO, biases_output, sizeof(biases_output));
            terminal_log("...DONE");
        }
        application_state(AS_READY);
        break;
	case AS_DRAWING:
        display_menu(drawing_menu_texts, ARRAY_SIZE(drawing_menu_texts));
        draw_and_predict();
        application_state(AS_READY);
        break;
    }
}

/*
 * Main application loop logic
 */
void main_loop()
{
    char ch;
	switch (TheApplication.state) {
	case AS_READY:
		if (kbhit()) {
			ch = getch();
            switch (ch) {
	            case PETSCII_F1:
                application_state(AS_TRAINING);
                break;
	            case PETSCII_F3:
                application_state(AS_SAVING);
                break;
	            case PETSCII_F5:
                application_state(AS_LOADING);
                break;
	            case PETSCII_F7:
                application_state(AS_DRAWING);
                break;
		    }
        }
		break;
	case AS_TRAINING:
		if (kbhit()) {
			ch = getch();
		}
		break;
    }
}

int main(void)
{
    // Install trampoline
    mmap_trampoline();
	
    // make all of RAM visible to the CPU
	mmap_set(MMAP_RAM);
	
    // place custom font
	memcpy(Charset, charset, 2048);
    
    // copy background (screen codes)
    memcpy(Screen, petmate_screen, 1000);
    
    // Hide BASIC, we aren't going to use it anyway
    mmap_set(MMAP_NO_BASIC);
    
    // Setup VIC to point to the new screen memory and charset
    vic_setmode(VICM_TEXT, Screen, Charset);
    
    // Copy background color data *after* setting vic mode
    memcpy(Color, petmate_color, 1000);
	
    // Set Background and border color
	vic.color_border = VCOL_BLACK;
	vic.color_back = VCOL_BLACK;
    
    // Sprites data setup
    spr_init(Screen);
    oscar_expand_lzo(Sprite0, sprite_data);    
    spr_set(0, false, 304, 58, (unsigned)Sprite0 / 64, VCOL_LT_GREEN, false, true, true);
    spr_set(1, false, 8+24, 58, ((unsigned)Sprite0 / 64) + 1, VCOL_LT_RED, false, false, false);

    // Windows initialization
    cwin_init(&cw_terminal, Screen, TERMINAL_LEFT, TERMINAL_TOP, TERMINAL_WIDTH, TERMINAL_HEIGHT);
    cwin_init(&cw_menu, Screen, MENU_LEFT, MENU_TOP, MENU_WIDTH, MENU_HEIGHT);
    cwin_init(&cw_canvas, Screen, CANVAS_LEFT, CANVAS_TOP, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    application_state(AS_READY);
	
    for(;;) {
		main_loop();
    }

    // Exit program, restoring default memory mapping
	// restore VIC
	vic_setmode(VICM_TEXT, (char *)0x0400, (char *)0x1000);
    // Reenable BASIC
    mmap_set(MMAP_ROM);
    return 0;
}
