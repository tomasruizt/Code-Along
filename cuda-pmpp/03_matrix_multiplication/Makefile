NVCC = nvcc
CFLAGS = -g
LIBS = -larmadillo

OBJECTS = matmul.o

program: $(OBJECTS) main1.cu
	$(NVCC) $(CFLAGS) $(OBJECTS) main1.cu -o program $(LIBS)

%.o: %.cu %.cuh
	$(NVCC) $(CFLAGS) -c $< -o $@

run: program
	echo "running..." && ./program

profile: program
	ncu --set full ./program > profile.txt

clean:
	rm -f program $(OBJECTS)