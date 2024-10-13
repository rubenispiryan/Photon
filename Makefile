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
	python3 photon.py sim ./$(OUT).phtn

com: photon.py ./$(OUT).phtn
	python3 photon.py com ./$(OUT).phtn

test: photon.py test.py
	mypy --disallow-untyped-defs photon.py
	mypy --disallow-untyped-defs test.py
	python3 test.py

snap: photon.py test.py
	python3 test.py --snapshot