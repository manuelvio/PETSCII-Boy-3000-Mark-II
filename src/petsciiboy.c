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
#include <c64/rasterirq.h>
#include "neuralnet.h"
#include "batch.h"

/*
MIT License

Copyright (c) 2025-Present Manuel Vio

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, i++
ncluding without limitation the rights
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
#pragma region(sprites, 0xb700, 0xb800, , , {sprites})
#pragma data(sprites)

const char sprite_data[] = {
	#embed spd_sprites lzo "../resources/neural_sprites.spd"
};


#pragma data(data)

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define CLAMP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))
#define ROUND_TO_INT8(x) (((x) >= 0) ? (int8_t)((x) + 0.5) : (int8_t)((x) - 0.5))

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

#define CANVAS_PIXEL_OFF    ' '
#define CANVAS_PIXEL_ON     '*'

#define BATCHES_COUNT 16

CharWin cw_menu;
CharWin cw_terminal;
CharWin cw_canvas;

RIRQCode	frame_rirq;

// Buffer used for terminal output
char terminal_buf[37];
char raster_buf[5];
char raster_buf2[5];

// Whenever a key is pressed during an interrupt check its value is stored here
char last_pressed_key;

// PETSCII char codes of nine histogram levels, ordered from lowest (0) to higher (8)
static const char activation_histogram_levels[9] = { 32, 100, 111, 121, 98, 248, 247, 227, 224 };

// Game state
enum ApplicationState {
	AS_READY,			// Getting ready
	AS_TRAINING,		// Network is training
    AS_ACCURACY_CHECK,  // Network is checking its parameters
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
	ApplicationState	state;		     // Main application state
    uint8_t             epochs_left;     // Epochs to be processed
    uint8_t             batches_left;
    NeuralNetwork       neural_network;  // Our neural network parameters
}	TheApplication;

Training training;

static const char * main_menu_texts[] = {
  "F1-TRAIN",
  "F3-SAVE PARAMS",
  "F5-LOAD PARAMS",
  "F7-DRAW DIGIT",
  "F8-VERIFY ACCURACY"
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

/*
 * Polls the keyboard and save the code of pressed key
 * If the last key has not been processed yet, skips the check
 * 
 * This function is intended for use inside an interrupt driven loop
 */
void keypressed(void)
{
	if (!last_pressed_key) {
        // Last pressed key has already been processed
        // or no key has been pressed yet, so we can poll the keyboard
		keyb_poll();
        // If a key is being pressed, save its value
		if (keyb_key & KSCAN_QUAL_DOWN) last_pressed_key = keyb_key;
	}
}

// Forward declaration
void application_state(ApplicationState state);

/*
 * Displays main menu data
 */
void display_menu(CharWin* win, const char **menu, uint8_t items_count)
{
    cwin_fill_rect(win, 0, 0, win->wx, win->wy, ' ', MENU_COLOR);
    for (uint8_t entry_id = 0; entry_id < items_count; entry_id++) {
        cwin_putat_string(win, 0, entry_id, menu[entry_id], MENU_COLOR);
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
 * Scrolls up window and adds a new row of text
 */
void window_log(CharWin* win, const char *str)
{
    cwin_scroll_up(win, 1);
    cwin_fill_rect(win, 0, win->wy - 1, win->wx, 1, ' ', TERMINAL_COLOR);
    cwin_putat_string(win, 0, win->wy - 1, str, TERMINAL_COLOR);
}

/*
 * Display a message and wait for a Yes/No answer, returns true/false accordingly
 * In this case everything different from a 'Y' is false
 */
bool confirm(CharWin* win, const char * msg)
{
    window_log(win, msg);
    return getch() == 89;
}

/*
 * Copy an array of bytes representing a digit into sprite 0 memory
 */
void draw_digit(input_t input)
{
    uint8_t sprite_idx = 0;
    uint8_t input_idx = 0;
    for(uint8_t input_row = 0; input_row < 16; input_row++) {
        // For every input row copy 16 bit of data and increment references
        Sprite0[sprite_idx++] = input[input_idx++];
        Sprite0[sprite_idx++] = input[input_idx++];
        // Since a sprite row is made of 3 bytes and input row is 2 byte we skip to the next one
        sprite_idx++;
    }
}

/*
 * Draws a histogram using PETSCII characters starting at a specific screen coordinate
 * x and y arguments refer to screen coordinates
 * Note: despite its simplicity it's actually quite slow due to floating point operations
 * In this form it's not a good candidate to raster interrupt inclusion
 */
void petscii_histogram(uint8_t x, uint8_t y, float values[], uint8_t num_values)
{
    uint16_t screen_pos = y * 40 + x;
    for(uint8_t i = 0; i < num_values; i++) {
        Screen[screen_pos + i] = activation_histogram_levels[ROUND_TO_INT8(values[i] * 8.0)];
    }
}


/*
 * Main training loop, iterates over the input batches for the epochs number
 * Every batch is used for training and checked to compute overall network accuracy
 */
void train_loop(NeuralNetwork *neural_network, Training *training)
{
    init_training(training);
    init_network(neural_network);
    while(!training->stopped && training->batch_index > -1) {
        load_training_batch(DRIVE_NO, training);
        while(!training->stopped && training->record_index < training->loaded_records) {
            train(neural_network, training->batch[training->record_index], training->batch[training->record_index][BATCH_ROW_LENGTH - 1]);
            training->processed++;
            training->record_index++;
        }
        training->batch_index--;
    }
}

/*
 * Takes a random batch and checks every record against current network
 * to verify current accuracy
 */
void accuracy_loop(NeuralNetwork *neural_network, Training *training)
{
    init_training(training);
    load_training_batch(DRIVE_NO, training);
    while(!training->stopped && training->record_index < training->loaded_records) {
        uint8_t guessed_digit  = predict(neural_network, training->batch[training->record_index]);
        training->correct += training->batch[training->record_index][BATCH_ROW_LENGTH - 1] == guessed_digit;
        training->processed++;
        training->record_index++;
    }
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
void draw_and_predict(NeuralNetwork *neural_network)
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
                window_log(&cw_terminal, "INPUT MODE: JOYSTICK");
                break;
            case PETSCII_F3:
                input_mode = LIGHT_PEN;
                window_log(&cw_terminal, "INPUT MODE: LIGHTPEN");
                break;
            case PETSCII_F5:
                done = confirm(&cw_terminal, "EXIT DRAWING? (Y/N)");
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
    uint8_t predicted = predict(neural_network, current_input);
    petscii_histogram(19, 2, neural_network->activations_hidden, HIDDEN_LAYER_SIZE);
    petscii_histogram(19, 4, neural_network->activations_output, OUTPUT_LAYER_SIZE);
    display_char(predicted);
    sprintf(terminal_buf, "I THINK YOU WROTE A %d", predicted);
    window_log(&cw_terminal, terminal_buf);

    // Collecting user's feedback about prediction and adjusting parameters
    if (!confirm(&cw_terminal, "AM I RIGHT?")) {
        bool is_number = false;
        do {
            window_log(&cw_terminal, "ENTER THE CORRECT DIGIT (0-9)");
            char answer = getch();
            is_number = answer >= 48 && answer <= 57;
            if (is_number) predicted = answer - 48;
        } while (!is_number);
    }
    window_log(&cw_terminal, "ADJUSTING WEIGHTS...");
    train(neural_network, current_input, predicted);
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
        display_menu(&cw_menu, main_menu_texts, ARRAY_SIZE(main_menu_texts));
        break;
	case AS_TRAINING:
        display_menu(&cw_menu, training_menu_texts, ARRAY_SIZE(training_menu_texts));
        window_log(&cw_terminal, "IT WILL TAKE A LOT OF TIME");
        window_log(&cw_terminal, "MAKE A CUP OF TEA");
        window_log(&cw_terminal, "PUT A RECORD ON");
        spr_show(0, true);
        train_loop(&TheApplication.neural_network, &training);
        spr_show(0, false);            
        application_state(AS_READY);
        break;
    case AS_ACCURACY_CHECK:
        spr_show(0, true);
        accuracy_loop(&TheApplication.neural_network, &training);
        spr_show(0, false);
        sprintf(terminal_buf, "ACCURACY=%.2f%%", ((float)training.correct / training.processed) * 100.0);
        window_log(&cw_terminal, terminal_buf);
        application_state(AS_READY);
        break;
	case AS_SAVING:
        cwin_fill_rect(&cw_menu, 0, 0, cw_menu.wx, cw_menu.wy, ' ', MENU_COLOR);
        if (confirm(&cw_terminal, "SAVE PARAMETERS? (Y/N)")) {
            window_log(&cw_terminal, "SAVING...");
            save_bytes("@0:WH,U,W", DRIVE_NO, TheApplication.neural_network.weights_hidden, sizeof(TheApplication.neural_network.weights_hidden));
            save_bytes("@0:WO,U,W", DRIVE_NO, TheApplication.neural_network.weights_output, sizeof(TheApplication.neural_network.weights_output));
            save_bytes("@0:BH,U,W", DRIVE_NO, TheApplication.neural_network.biases_hidden, sizeof(TheApplication.neural_network.biases_hidden));
            save_bytes("@0:BO,U,W", DRIVE_NO, TheApplication.neural_network.biases_output, sizeof(TheApplication.neural_network.biases_output));
            window_log(&cw_terminal, "...DONE");
        }
        application_state(AS_READY);
        break;
	case AS_LOADING:
        cwin_fill_rect(&cw_menu, 0, 0, cw_menu.wx, cw_menu.wy, ' ', MENU_COLOR);
        if (confirm(&cw_terminal, "LOAD PARAMETERS? (Y/N)")) {
            window_log(&cw_terminal, "LOADING...");
            load_bytes("WH,U,R", DRIVE_NO, TheApplication.neural_network.weights_hidden, sizeof(TheApplication.neural_network.weights_hidden));
            load_bytes("WO,U,R", DRIVE_NO, TheApplication.neural_network.weights_output, sizeof(TheApplication.neural_network.weights_output));
            load_bytes("BH,U,R", DRIVE_NO, TheApplication.neural_network.biases_hidden, sizeof(TheApplication.neural_network.biases_hidden));
            load_bytes("BO,U,R", DRIVE_NO, TheApplication.neural_network.biases_output, sizeof(TheApplication.neural_network.biases_output));
            window_log(&cw_terminal, "...DONE");
        }
        application_state(AS_READY);
        break;
	case AS_DRAWING:
        display_menu(&cw_menu, drawing_menu_texts, ARRAY_SIZE(drawing_menu_texts));
        draw_and_predict(&TheApplication.neural_network);
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
	            case PETSCII_F8:
                application_state(AS_ACCURACY_CHECK);
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

/*
 * This function is called at every frame using a raster interrupt
 */
__interrupt void frame_irq(void)
{
	vic.color_border++;
    static uint16_t last_processed = UINT16_MAX; // Keep track of last processed batch item 
	switch (TheApplication.state) {
        case AS_TRAINING:
        case AS_ACCURACY_CHECK:
            // We're training the network or verifying its accuracy, if RUN/STOP is pressed flag "stopped" in Training structure is set
            // While looping the flag is checked at every step, if it's set the loop is interrupted (pun not intended)
            keypressed();
            if (last_pressed_key & KSCAN_QUAL_DOWN) {
                if ((last_pressed_key & KSCAN_QUAL_MASK) == KSCAN_STOP) training.stopped = true;
                last_pressed_key = 0;
            }

            // Draw currently processed digit and display some info,
            // but only if it changed since last frame
            if (training.processed != last_processed) {
                draw_digit(training.batch[training.record_index]);

                itoa(training.batch_index, raster_buf, 10);
                itoa(training.record_index, raster_buf2, 10);
                cwin_putat_string(&cw_menu, 0, 6, raster_buf, MENU_COLOR);
                cwin_putat_string(&cw_menu, 0, 7, raster_buf2, MENU_COLOR);
                last_processed = training.processed;        
            }
            break;
        default:
            break;
    }
    vic.color_border--;
}

int main(void)
{
    // Fixed random seed to simplify debugging
    srand(74);

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
    spr_set(1, false, 8 + 24, 58, ((unsigned)Sprite0 / 64) + 1, VCOL_LT_RED, false, false, false);

    // Windows initialization
    cwin_init(&cw_terminal, Screen, TERMINAL_LEFT, TERMINAL_TOP, TERMINAL_WIDTH, TERMINAL_HEIGHT);
    cwin_init(&cw_menu, Screen, MENU_LEFT, MENU_TOP, MENU_WIDTH, MENU_HEIGHT);
    cwin_init(&cw_canvas, Screen, CANVAS_LEFT, CANVAS_TOP, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    application_state(AS_READY);

    // Init the raster IRQ system to use the kernal iterrupt vector
	rirq_init_kernal();

	// Init the music interrupt on raster line 250
	rirq_build(&frame_rirq, 1);
	rirq_call(&frame_rirq, 0, frame_irq);
	rirq_set(0, 50, &frame_rirq);

	// Prepare the raster IRQ order
	rirq_sort();

	// start raster IRQ processing
	rirq_start();



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
