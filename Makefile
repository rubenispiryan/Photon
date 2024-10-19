.SILENT:

INPUT ?= output

.PHONY: run

run: output
	./output

output: output.o
	ld -o output output.o \
        -lSystem \
        -syslibroot `xcrun -sdk macosx --show-sdk-path` \
        -e _start \
        -arch arm64

output.o: com
	as -g -o output.o output.s

sim: photon.py ./$(INPUT).phtn
	pypy3.10 photon.py sim ./$(INPUT).phtn

com: photon.py ./$(INPUT).phtn
	pypy3.10 photon.py com ./$(INPUT).phtn

test: photon.py test.py
	mypy --disallow-untyped-defs photon.py
	mypy --disallow-untyped-defs test.py
	pypy3.10 test.py

snap: photon.py test.py
	pypy3.10 test.py --snapshot

clean:
	rm output output.o output.s