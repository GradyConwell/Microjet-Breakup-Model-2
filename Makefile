# Compiler and environment paths
MICROMAMBA_ENV ?= $(HOME)/.micromamba/envs/heyoka-env
CXX := micromamba run -r $(HOME)/.micromamba -n heyoka-env arm64-apple-darwin20.0.0-clang++

# Flags
CXXFLAGS ?= -std=c++23 -O2 -Wall -Wextra -Wno-pragma-clang-attribute \
            -I$(MICROMAMBA_ENV)/include
LDFLAGS ?= -L$(MICROMAMBA_ENV)/lib -lheyoka -Wl,-rpath,$(MICROMAMBA_ENV)/lib

# Target and sources
TARGET = microjet
SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run the compiled executable
run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(OBJS) $(TARGET) build/
