# Compiler and environment paths
MICROMAMBA_ENV ?= $(HOME)/.micromamba/envs/heyoka-env

ifeq ($(OS),Windows_NT)
    # Native Windows (conda/micromamba environment with clang++ and Make)
    ENV_PREFIX ?= $(if $(CONDA_PREFIX),$(CONDA_PREFIX)/Library,$(MICROMAMBA_ENV)/Library)
    CXX ?= clang++
    CXXFLAGS ?= -std=c++23 -O2 -Wall -Wextra -Wno-pragma-clang-attribute \
                -D_CRT_SECURE_NO_WARNINGS -I"$(ENV_PREFIX)/include"
    LDFLAGS ?= -L"$(ENV_PREFIX)/lib" -lheyoka
    TARGET = microjet.exe
    RUN_CMD = $(TARGET)
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        CXX := micromamba run -r $(HOME)/.micromamba -n heyoka-env arm64-apple-darwin20.0.0-clang++
    else
        # Linux / WSL2
        CXX := micromamba run -r $(HOME)/.micromamba -n heyoka-env clang++
    endif
    CXXFLAGS ?= -std=c++23 -O2 -Wall -Wextra -Wno-pragma-clang-attribute \
                -I$(MICROMAMBA_ENV)/include
    LDFLAGS ?= -L$(MICROMAMBA_ENV)/lib -lheyoka -Wl,-rpath,$(MICROMAMBA_ENV)/lib
    TARGET = microjet
    RUN_CMD = ./$(TARGET)
endif

# Sources and objects
SRCS = $(wildcard *.cpp)
HDRS = $(wildcard *.hpp)
OBJS = $(SRCS:.cpp=.o)

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run the compiled executable
run: $(TARGET)
	$(RUN_CMD)

clean:
	rm -rf $(OBJS) microjet microjet.exe build/
