#ifndef PB_LIGHT_PEN_H
#define PB_LIGHT_PEN_H

#include <stdint.h>

#define Screen	((char *)0xcc00)
#define Charset	((char *)0xd000)

#pragma compile("light_pen.c")

#endif