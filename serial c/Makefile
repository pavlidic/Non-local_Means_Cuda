SHELL := /bin/bash

CC=gcc

CFLAGS = -O3
LINK = -lm

all: serial

%.o: %.c
	$(CC) $(CFLAGS) -c $^



serial: serial.o extra.o
	$(CC) $(CFLAGS) -o $@ $^ $(LINK)
