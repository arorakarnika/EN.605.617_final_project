NVCC = nvcc
CFLAGS = -O2 -arch=sm_75 --ptxas-options=-v -Iinclude
TARGET = vocab_lookup.exe

SRC_DIR = src
INCLUDE_DIR = include
DATA_DIR = data

SOURCES = $(SRC_DIR)/vocab_lookup.cu

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) benchmark_results.csv $(DATA_DIR)/sample_vocab.bin

.PHONY: all clean
