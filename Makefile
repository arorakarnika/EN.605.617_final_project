NVCC = nvcc
CFLAGS = -O2 -arch=sm_75 --ptxas-options=-v -Iinclude
TARGET = bpe_tokenizer.exe

SRC_DIR = src
INCLUDE_DIR = include

SOURCES = \
	$(SRC_DIR)/main.cu \
	$(SRC_DIR)/bpe_kernels.cu \
	$(SRC_DIR)/bpe_io.cu \
	$(SRC_DIR)/bpe_benchmark.cu

HEADERS = $(INCLUDE_DIR)/bpe.h

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)

.PHONY: all clean
