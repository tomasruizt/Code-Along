COMPILE_CMD = nvcc -g -o program main.cu

run: compile
	echo "running..." && ./program

compile:
	echo "compiling..." && $(COMPILE_CMD)

debug_compile:
	echo "compiling..." && $(COMPILE_CMD) -g -G

profile:
	$(COMPILE_CMD) && ncu --set full ./program > profile.txt
