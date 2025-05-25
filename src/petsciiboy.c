#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <c64/memmap.h>
#include <c64/vic.h>

/*
MIT License

Copyright (c) 2025 Manuel Vio

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

// Load custom charset at 0xc000, we'll copy it later where needed (0xd000) and reuse this memory area
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

#pragma data(data)

#define Screen	((char *)0xcc00)
#define Charset	((char *)0xd000)
#define Color	((char *)0xd800)

int main(void) {
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
 
	for(;;) {
        
    }

    // Exit program, restoring default memory mapping
	// restore VIC
	vic_setmode(VICM_TEXT, (char *)0x0400, (char *)0x1000);
    // Reenable BASIC
    mmap_set(MMAP_ROM);
    return 0;
}
