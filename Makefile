.SILENT:

INPUT ?= output

.PHONY: run

run: com output
	./output

output: output.o
	ld -o output output.o

output.o: output.s
	as -o output.o output.s

sim: photon.py ./$(INPUT).phtn
	pypy3.10 photon.py sim ./$(INPUT).phtn

flow: photon.py ./$(INPUT).phtn
	pypy3.10 photon.py flow ./$(INPUT).phtn

com: photon.py ./$(INPUT).phtn
	pypy3.10 photon.py com ./$(INPUT).phtn

test: photon.py test.py
	mypy --disallow-untyped-defs photon.py
	mypy --disallow-untyped-defs test.py
	pypy3.10 test.py

snap: photon.py test.py
	pypy3.10 test.py --snapshot

dot: flow ./$(INPUT).dot
	dot -Tsvg $(INPUT).dot -o $(INPUT).svg

clean:
	rm output output.o output.s output.svg output.dot