NVCC = nvcc
CFLAGS = -g -G -lineinfo
LIBS = -larmadillo

OBJECTS = matmul.o

program1: $(OBJECTS) main1.cu
	$(NVCC) $(CFLAGS) $(OBJECTS) main1.cu -o program1 $(LIBS)

program2: $(OBJECTS) main2.cu
	$(NVCC) $(CFLAGS) $(OBJECTS) main2.cu -o program2

%.o: %.cu %.cuh
	$(NVCC) $(CFLAGS) -c $< -o $@

run1: program1
	echo "running program1..." && ./program1

run2: program2
	echo "running program2..." && ./program2

profile: program1
	ncu --set full ./program1 > profile.txt

profile2: program2
	ncu --set full ./program2 > profile2.txt

nsight_profile2: program2
	nsys profile -o profiles/program2 ./program2

ncu_profile2: program2
	ncu --set full --kernel-name matrix_mult --launch-skip 0 --launch-count 1 -o profiles/matrix_mult ./program2

clean:
	rm -f program1 program2 $(OBJECTS)