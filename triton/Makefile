.PHONY: help profile-rms profile-kv

help:
	@echo "Targets:"
	@echo "  profile-rms   Profile Triton rms_norm_kernel with ncu"
	@echo "  profile-kv    Profile Triton kv_lookup_kernel with ncu"

profile-rms:
	mkdir -p ncu/
	ncu -k rms_norm_kernel python rms_norm.py 2>&1 > ncu/rms_norm.txt

profile-kv:
	mkdir -p ncu/
	ncu -k kv_lookup_kernel python paged_kv_gather.py 2>&1 > ncu/kv_lookup.txt

