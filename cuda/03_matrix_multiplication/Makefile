COMPILE_CMD = nvcc -g -o program main.cu -larmadillo

run:
	$(COMPILE_CMD) && ./program

profile:
	$(COMPILE_CMD) && ncu --set full ./program > profile.txt