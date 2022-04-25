CC = gcc
CXX = g++
CFLAGS = -std=c99 -ggdb -O0 -I../common
LIBS = -lm -lOpenCL

INCLUDES = imagenet_labels.h

all: vgg

vgg: main.o
<<<<<<< HEAD
	$(CC) main.o common/libutils.a -o vgg $(LIBS)
=======
	$(CC) main.o ./common/libutils.a -o vgg $(LIBS)
>>>>>>> f359f197ab08dcb3ad6ce9d958932bbe57b62ace

main.o: main.c $(INCLUDES)
	$(CC) -c $(CFLAGS) main.c

clean:
	rm *.o vgg
