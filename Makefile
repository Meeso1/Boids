all: boids
boids: boids.cu
	nvcc -o boids boids.cu
.PHONY: clean all
clean:
	rm boids