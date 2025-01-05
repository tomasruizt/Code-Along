COMPILE_CMD = nvcc -o program main.cu

run: compile
	./program

compile:
	$(COMPILE_CMD)

debug_compile:
	$(COMPILE_CMD) -g -G

profile:
	$(COMPILE_CMD) -g && ncu --set full ./program > profile.txt
