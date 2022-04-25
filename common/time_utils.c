#include "time_utils.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

static struct hsearch_data g_htab;
static bool is_initialised = false;
#define MAX_KEY_LEN 256


void time_measure_start(const char *key)
{
    if (!is_initialised)
    {
        is_initialised = true;
        hcreate_r(HASH_TABLE_SIZE, &g_htab);
    }

    struct timespec *start_time = (struct timespec *)malloc(sizeof(struct timespec));
    clock_gettime(CLOCK_REALTIME, start_time);

    size_t key_length = strnlen(key, MAX_KEY_LEN);
    char *key2 = calloc(key_length + 1, sizeof(char));
    strncpy(key2, key, key_length);

    ENTRY entry = {
        .key = key2,
        .data = (void *)start_time
    };

    ENTRY *result;

    if (hsearch_r(entry, FIND, &result, &g_htab))
    {
        result->data = (void *)start_time;
    }
    else
    {
        ENTRY * result;
        if (!hsearch_r(entry, ENTER, &result, &g_htab))
        {
            fprintf(stderr, "ERROR: Failed to register time measure, duplicate name?\n");
            abort();
        }
    }
}

double time_measure_stop(const char *key)
{
    if (!is_initialised)
    {
        fprintf(stderr, "ERROR: Not initialised, first start some timers.\n");
        abort();
    }

    size_t key_length = strnlen(key, MAX_KEY_LEN);
    char *key2 = calloc(key_length + 1, sizeof(char));
    strncpy(key2, key, key_length);

    ENTRY query = {
        .key = key2,
        .data = NULL
    };
    ENTRY *result;

    if (!hsearch_r(query, FIND, &result, &g_htab))
    {
        fprintf(stderr, "ERROR: timestamp not found.\n");
        abort();
    }

    struct timespec *start_time = (struct timespec *)result->data;
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    return (double)(now.tv_sec - start_time->tv_sec) +
            ((double)(now.tv_nsec - start_time->tv_nsec) * 1e-9);
}

void time_measure_stop_and_print(const char *key)
{
    double elapsed_time = time_measure_stop(key);
    printf("Elapsed time for %s is %lf s\n", key, elapsed_time);
}
