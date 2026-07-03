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
	$(MODAL_RUN) --script 01_stream_basics.py --name stream-basics --n-procs 1
	modal volume get --force $(VOLUME) nsys/stream-basics.nsys-rep $(TRACE_DIR)/
	rm -f $(TRACE_DIR)/stream-basics.sqlite

exercise-02: traces
	$(MODAL_RUN) --script 02_copy_compute_overlap.py --name copy-compute-overlap --n-procs 1
	modal volume get --force $(VOLUME) nsys/copy-compute-overlap.nsys-rep $(TRACE_DIR)/
	rm -f $(TRACE_DIR)/copy-compute-overlap.sqlite

exercise-03: traces
	$(MODAL_RUN) --script 03_manual_async_allreduce.py --name async-allreduce --n-procs 2
	modal volume get --force $(VOLUME) nsys/async-allreduce-rank0.nsys-rep $(TRACE_DIR)/
	modal volume get --force $(VOLUME) nsys/async-allreduce-rank1.nsys-rep $(TRACE_DIR)/
	rm -f $(TRACE_DIR)/async-allreduce-rank0.sqlite $(TRACE_DIR)/async-allreduce-rank1.sqlite

exercise-04: traces
	$(MODAL_RUN) --script 04_tiny_ddp_bucket_overlap.py --name tiny-ddp-buckets --n-procs 2
	modal volume get --force $(VOLUME) nsys/tiny-ddp-buckets-rank0.nsys-rep $(TRACE_DIR)/
	modal volume get --force $(VOLUME) nsys/tiny-ddp-buckets-rank1.nsys-rep $(TRACE_DIR)/
	rm -f $(TRACE_DIR)/tiny-ddp-buckets-rank0.sqlite $(TRACE_DIR)/tiny-ddp-buckets-rank1.sqlite
