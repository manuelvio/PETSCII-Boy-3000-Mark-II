#ifndef PB_STRINGUTILS_H
#define PB_STRINGUTILS_H

#include <stdint.h>

static const char hex_chars[] = "0123456789ABCDEF";

/*
 * Converts a byte to its hex representation
 * Avoids the use of sprintf for such a trivial task
 */
void byte_to_hex(uint8_t byte, char *hex_string);

#pragma compile("stringutils.c")

#endif