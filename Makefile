.SILENT:

OUT ?= output

.PHONY: run

run: $(OUT)
	./$(OUT)

$(OUT): $(OUT).o
	ld -o $(OUT) $(OUT).o \
        -lSystem \
        -syslibroot `xcrun -sdk macosx --show-sdk-path` \
        -e _start \
        -arch arm64

$(OUT).o: com
	as -g -o $(OUT).o $(OUT).s

sim: photon.py ./$(OUT).phtn
	pypy3.10 photon.py sim ./$(OUT).phtn

com: photon.py ./$(OUT).phtn
	pypy3.10 photon.py com ./$(OUT).phtn

test: photon.py test.py
	mypy --disallow-untyped-defs photon.py
	mypy --disallow-untyped-defs test.py
	pypy3.10 test.py

snap: photon.py test.py
	pypy3.10 test.py --snapshot