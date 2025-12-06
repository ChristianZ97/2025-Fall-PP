CC = gcc
NVCC = nvcc
HIPCC = hipcc

CFLAGS = -Wall -g
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -DPROFILING
LDFLAGS = -lm
HIPCCFLAGS = -std=c++14 -O3 --offload-arch=gfx908 -DPROFILING

# 1. 自動抓取當前目錄下所有的 .cu 檔案 (例如: hw4_64_64.cu, hw4.cu)
CU_SOURCES = $(wildcard *.cu)

# 2. 將 .cu 副檔名移除，作為執行檔名稱 (例如: hw4_64_64.cu -> hw4_64_64)
CU_EXECS = $(patsubst %.cu,%,$(CU_SOURCES))

# 定義所有要生成的目標：包含 C 語言程式與所有的 CUDA 執行檔
TARGETS = $(CU_EXECS)

all: $(TARGETS)

clean:
	rm -f $(TARGETS) hw4-amd

.PHONY: all clean

# 3. 通用規則：所有的 CUDA 執行檔 (%) 都由對應的 .cu 檔案 (%.cu) 編譯而來
# $< 代表第一個依賴檔案 (即 source code)
%: %.cu
	$(NVCC) $(NVFLAGS) -o $@ $< $(LDFLAGS)

# AMD HIP 維持原本設定 (若有需要也可以改成自動抓取 .hip)
hw4-amd: hw4.hip
	$(HIPCC) $(HIPCCFLAGS) -o $@ $< $(LDFLAGS)
