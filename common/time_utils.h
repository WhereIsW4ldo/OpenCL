#ifndef __TIME_UTILS_H__
#define __TIME_UTILS_H__

#define _GNU_SOURCE
#include <search.h>

#define HASH_TABLE_SIZE 30

void time_measure_start(const char *key);
double time_measure_stop(const char *key);
void time_measure_stop_and_print(const char *key);

#endif // __TIME_UTILS_H__
