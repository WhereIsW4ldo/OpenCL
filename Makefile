CC = gcc
CXX = g++
CFLAGS = -std=c99 -ggdb -I../common
LIBS = -lm -lOpenCL

INCLUDES = imagenet_labels.h

all: vgg

vgg: main.o
	$(CC) main.o ./common/libutils.a -o vgg $(LIBS)

main.o: main.c $(INCLUDES)
	$(CC) -c $(CFLAGS) main.c

clean:
	rm *.o vgg
