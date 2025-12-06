HIPCC = hipcc

# HIP 編譯參數
# 注意：--offload-arch=gfx908 是針對 AMD MI100 GPU，若是 MI200/MI250 請改為 gfx90a
HIPCCFLAGS = -std=c++14 -O3 --offload-arch=gfx908 -DPROFILING
LDFLAGS = -lm

# 1. 自動抓取當前目錄下所有的 .hip 檔案 (例如: hw4_64_64.hip)
HIP_SOURCES = $(wildcard *.hip)

# 2. 將 .hip 副檔名移除，作為執行檔名稱 (例如: hw4_64_64.hip -> hw4_64_64)
HIP_EXECS = $(patsubst %.hip,%,$(HIP_SOURCES))

# 定義所有要生成的目標
TARGETS = $(HIP_EXECS)

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean

# 3. 通用規則：所有的 HIP 執行檔 (%) 都由對應的 .hip 檔案 (%.hip) 編譯而來
# $< 代表第一個依賴檔案 (即 source code)
%: %.hip
	$(HIPCC) $(HIPCCFLAGS) -o $@ $< $(LDFLAGS)
