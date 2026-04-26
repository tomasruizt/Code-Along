PYTHON ?= python
NCU    ?= ncu

# test.py runs 6 correctness shapes per kernel before the benchmark loop;
# skip those so the profile lands on a steady-state 4096x4096 launch.
SKIP   ?= 6
COUNT  ?= 4

NCU_FLAGS = --target-processes all --set full \
            --launch-skip $(SKIP) --launch-count $(COUNT)

.PHONY: test profile profile-naive profile-smem

test:
	$(PYTHON) test.py

profile: profile-naive profile-smem

profile-naive:
	$(NCU) $(NCU_FLAGS) --kernel-name naive_kernel \
	       $(PYTHON) test.py | tee profile_naive.txt

profile-smem:
	$(NCU) $(NCU_FLAGS) --kernel-name smem_kernel \
	       $(PYTHON) test.py | tee profile_smem.txt
