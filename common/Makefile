CC = gcc
CFLAGS = -Wall -std=c99 -ggdb -I../common

libutils.a: time_utils.o ocl_utils.o
	ar rsc libutils.a time_utils.o ocl_utils.o

time_utils.o: time_utils.c time_utils.h
	$(CC) $(CFLAGS) -c time_utils.c

ocl_utils.o: ocl_utils.c ocl_utils.h
	$(CC) $(CFLAGS) -c ocl_utils.c

