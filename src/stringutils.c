#include "stringutils.h"

void byte_to_hex(uint8_t byte, char *hex_string)
{
    hex_string[0] = hex_chars[(byte >> 4) & 0x0F];
    hex_string[1] = hex_chars[byte & 0x0F];
    hex_string[2] = '\0';
}
