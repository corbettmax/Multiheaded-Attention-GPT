# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -march=native -ffast-math -fopenmp -ftree-vectorize
LDFLAGS = -fopenmp

# Directories
SRC_DIR = src
BUILD_DIR = build
TARGET = attention_mechanism

# Source files
SOURCES = $(SRC_DIR)/main.cpp \
          $(SRC_DIR)/attentionmechanism.cpp \
          $(SRC_DIR)/multiheadedgpt.cpp \
          $(SRC_DIR)/util.cpp

# Object files
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Header dependencies
HEADERS = $(SRC_DIR)/attentionmechanism.hpp \
          $(SRC_DIR)/multiheadedgpt.hpp \
          $(SRC_DIR)/util.hpp

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

# Compile source files to object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# Rebuild everything
rebuild: clean all

# Run the program
run: $(TARGET)
	./$(TARGET)

# Phony targets
.PHONY: all clean rebuild run
