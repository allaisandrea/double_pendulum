# Builds the camera helper tools into bin/ (git-ignored).
BIN := bin
UVC_SRC := $(wildcard uvc-util/src/*.m)

all: $(BIN)/uvc-util $(BIN)/avprobe

$(BIN)/uvc-util: $(UVC_SRC) $(wildcard uvc-util/src/*.h) | $(BIN)
	cc -o $@ -framework IOKit -framework Foundation $(UVC_SRC)

$(BIN)/avprobe: avprobe/avprobe.swift | $(BIN)
	swiftc -O -o $@ $<

$(BIN):
	mkdir -p $@

clean:
	rm -rf $(BIN)

.PHONY: all clean
