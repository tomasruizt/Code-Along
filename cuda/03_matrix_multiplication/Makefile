COMPILE_CMD = nvcc -g -o program main.cu -larmadillo

run:
	echo "compiling..." && $(COMPILE_CMD) && echo "running..." && ./program

profile:
	$(COMPILE_CMD) && ncu --set full ./program > profile.txt