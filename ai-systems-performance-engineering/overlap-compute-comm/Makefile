SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

VOLUME := overlap-compute-comm
TRACE_DIR := traces
MODAL_RUN := $(if $(GPU),GPU=$(GPU) ,)modal run modal/modal_overlap.py

.PHONY: all profile-all exercise-01 exercise-02 exercise-03 exercise-04 traces list-traces

all: profile-all

profile-all: exercise-01 exercise-02 exercise-03 exercise-04

traces:
	mkdir -p $(TRACE_DIR)

list-traces:
	modal volume ls $(VOLUME) nsys

exercise-01: traces
	$(MODAL_RUN) --script 01_stream_basics.py --name 01_stream_basics --n-procs 1 2>&1 | tee $(TRACE_DIR)/01_stream_basics_stout.txt
	modal volume get --force $(VOLUME) nsys/01_stream_basics.nsys-rep $(TRACE_DIR)/
	rm -f $(TRACE_DIR)/01_stream_basics.sqlite

exercise-02: traces
	$(MODAL_RUN) --script 02_copy_compute_overlap.py --name 02_copy_compute_overlap --n-procs 1 2>&1 | tee $(TRACE_DIR)/02_copy_compute_overlap_stout.txt
	modal volume get --force $(VOLUME) nsys/02_copy_compute_overlap.nsys-rep $(TRACE_DIR)/
	rm -f $(TRACE_DIR)/02_copy_compute_overlap.sqlite

exercise-03: traces
	$(MODAL_RUN) --script 03_manual_async_allreduce.py --name 03_manual_async_allreduce --n-procs 2 2>&1 | tee $(TRACE_DIR)/03_manual_async_allreduce_stout.txt
	modal volume get --force $(VOLUME) nsys/03_manual_async_allreduce-rank0.nsys-rep $(TRACE_DIR)/
	modal volume get --force $(VOLUME) nsys/03_manual_async_allreduce-rank1.nsys-rep $(TRACE_DIR)/
	rm -f $(TRACE_DIR)/03_manual_async_allreduce-rank0.sqlite $(TRACE_DIR)/03_manual_async_allreduce-rank1.sqlite

exercise-04: traces
	$(MODAL_RUN) --script 04_tiny_ddp_bucket_overlap.py --name 04_tiny_ddp_bucket_overlap --n-procs 2 2>&1 | tee $(TRACE_DIR)/04_tiny_ddp_bucket_overlap_stout.txt
	modal volume get --force $(VOLUME) nsys/04_tiny_ddp_bucket_overlap-rank0.nsys-rep $(TRACE_DIR)/
	modal volume get --force $(VOLUME) nsys/04_tiny_ddp_bucket_overlap-rank1.nsys-rep $(TRACE_DIR)/
	rm -f $(TRACE_DIR)/04_tiny_ddp_bucket_overlap-rank0.sqlite $(TRACE_DIR)/04_tiny_ddp_bucket_overlap-rank1.sqlite
